from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple
import csv
import random

import numpy as np

from communication.latency_model import CV2XLatencyModel, summarize_round_latencies
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, auto_lambda_smooth
from simulation.config import MRSUSimulationConfig
from simulation.coverage import CoverageModel
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, local_rsu_chr_from_counts, summarize_metrics

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


METHOD_DQN = "dqn"
METHOD_LABEL = "DQN"
METHOD_NOTE = (
    "Conventional discrete DQN baseline. The agent observes mobility, coverage, "
    "hotspot geometry, historical-demand strength, and aggregate previous-round "
    "performance only. It does not observe predicted movie lists, demand_summary, "
    "dominant_contents, or vehicle-level miss-content feedback."
)

MRSU_CACHE_POLICIES = (
    "mrsu_current_history",
    "selected_hotspot_history",
    "mrsu_current_plus_hotspot_history",
    "global_history",
)
FRSU_CACHE_POLICIES = (
    "frsu_current_history",
    "frsu_only_history",
    "frsu_current_plus_frsu_only_history",
    "global_history",
)

GLOBAL_FEATURE_DIM = 13
CANDIDATE_FEATURE_DIM = 6


@dataclass
class DecodedAction:
    action_id: int
    hotspot_index: int
    mrsu_policy_index: int
    frsu_policy_index: int
    mrsu_cache_policy: str
    frsu_cache_policy: str

    def to_dict(self) -> dict:
        return {
            "action_id": int(self.action_id),
            "hotspot_index": int(self.hotspot_index),
            "mrsu_policy_index": int(self.mrsu_policy_index),
            "frsu_policy_index": int(self.frsu_policy_index),
            "mrsu_cache_policy": self.mrsu_cache_policy,
            "frsu_cache_policy": self.frsu_cache_policy,
        }


def state_dim(candidate_count: int) -> int:
    return GLOBAL_FEATURE_DIM + int(candidate_count) * CANDIDATE_FEATURE_DIM


def action_dim(candidate_count: int) -> int:
    return int(candidate_count) * len(MRSU_CACHE_POLICIES) * len(FRSU_CACHE_POLICIES)


def decode_action(action_id: int, candidate_count: int) -> DecodedAction:
    action_count = action_dim(candidate_count)
    action_id = int(action_id) % max(action_count, 1)
    policy_block = len(MRSU_CACHE_POLICIES) * len(FRSU_CACHE_POLICIES)
    hotspot_index = action_id // policy_block
    remainder = action_id % policy_block
    mrsu_policy_index = remainder // len(FRSU_CACHE_POLICIES)
    frsu_policy_index = remainder % len(FRSU_CACHE_POLICIES)
    return DecodedAction(
        action_id=action_id,
        hotspot_index=int(hotspot_index),
        mrsu_policy_index=int(mrsu_policy_index),
        frsu_policy_index=int(frsu_policy_index),
        mrsu_cache_policy=MRSU_CACHE_POLICIES[mrsu_policy_index],
        frsu_cache_policy=FRSU_CACHE_POLICIES[frsu_policy_index],
    )


class DQNBaselineScenario:
    """DQN MDP wrapper with training-history-only content decisions."""

    def __init__(
        self,
        config: MRSUSimulationConfig,
        rounds: int | None = None,
        seed_offset: int = 0,
        candidate_count: int | None = None,
        reward_not_covered_penalty: float = 0.0,
        reward_not_cached_penalty: float = 0.0,
        latency_model: CV2XLatencyModel | None = None,
    ):
        self.base_config = config
        self.config = replace(
            config,
            seed=int(config.seed) + int(seed_offset),
            rounds=int(rounds if rounds is not None else config.rounds),
        )
        self.rounds = int(self.config.decision_rounds)
        self.candidate_count = int(candidate_count if candidate_count is not None else config.candidate_count)
        self.reward_not_covered_penalty = float(reward_not_covered_penalty)
        self.reward_not_cached_penalty = float(reward_not_cached_penalty)
        self.latency_model = latency_model or CV2XLatencyModel()

        self.env: MRSUEnvironment | None = None
        self.hotspot_generator: CandidateHotspotGenerator | None = None
        self.planner: QPPathPlanner | None = None
        self.vehicle_history_demands: Dict[int, Counter] = {}
        self.current_hotspots: List[dict] = []
        self.current_state = np.zeros(state_dim(self.candidate_count), dtype=np.float32)
        self.round_index = 0

    @property
    def state_dim(self) -> int:
        return state_dim(self.candidate_count)

    @property
    def action_dim(self) -> int:
        return action_dim(self.candidate_count)

    def reset(self) -> np.ndarray:
        self.env = MRSUEnvironment(self.config)
        self.hotspot_generator = CandidateHotspotGenerator(
            road_length=self.config.road_length,
            grid_step=self.config.grid_step,
            mrsu_radius=self.config.mrsu_radius,
            mrsu_cache_capacity=self.config.mrsu_cache_capacity,
            candidate_count=self.candidate_count,
        )
        self.planner = QPPathPlanner(
            road_length=self.config.road_length,
            dt=self.config.dt,
            horizon=self.config.planner_horizon,
            v_min=self.config.mrsu_v_min,
            v_max=self.config.mrsu_v_max,
            a_min=self.config.mrsu_a_min,
            a_max=self.config.mrsu_a_max,
        )
        self.vehicle_history_demands = {
            int(vehicle_id): Counter(counter)
            for vehicle_id, counter in self.env.vehicle_prediction_profiles.items()
        }
        self.round_index = 0
        self._prepare_round()
        return self.current_state.copy()

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.env is None or self.planner is None:
            raise RuntimeError("Scenario must be reset before stepping.")

        decoded = decode_action(action_id, self.candidate_count)
        selected_hotspot = self._select_hotspot(decoded.hotspot_index)
        request_budget = max(float(self._history_strength(range(self.config.vehicle_num))), 1.0)
        lambda_smooth = auto_lambda_smooth(
            current_position=self.env.mrsu.position,
            target_position=float(selected_hotspot.get("position", self.env.mrsu.position)),
            potential_cache_gain=float(selected_hotspot.get("potential_cache_gain", 0.0)),
            road_length=self.config.road_length,
            request_budget=request_budget,
            default_lambda=self.config.default_lambda_smooth,
        )
        path_plan = self.planner.plan(
            current_position=self.env.mrsu.position,
            current_speed=self.env.mrsu.speed,
            target_position=float(selected_hotspot.get("position", self.env.mrsu.position)),
            lambda_smooth=lambda_smooth,
        )
        window_ticks = self.env.decision_window_ticks(self.round_index)
        decision_coverage = self.env.project_service_window_coverage(path_plan, ticks=window_ticks)

        mrsu_cache, frsu_cache = self._build_caches(decoded, decision_coverage, selected_hotspot)
        self.env.set_cache(mrsu_cache, frsu_cache)
        coverage = self.env.execute_service_window(path_plan, ticks=window_ticks)
        true_requests = self.env.sample_round_requests(self.round_index)
        metrics = self.env.evaluate(true_requests, coverage, mrsu_cache, frsu_cache)
        latency = self.latency_model.evaluate_round(
            vehicle_requests=true_requests,
            vehicle_positions=self.env.mobility.positions(),
            coverage=coverage,
            mrsu_position=self.env.mrsu.position,
            frsu_position=self.env.frsu.position,
            road_length=self.config.road_length,
            mrsu_radius=self.config.mrsu_radius,
            frsu_radius=self.config.frsu_radius,
            mrsu_cache=mrsu_cache,
            frsu_cache=frsu_cache,
        ).to_dict()
        reward = self._reward(metrics)

        info = {
            "round": int(self.round_index),
            "physical_tick_start": int(self.round_index * self.config.decision_interval),
            "physical_tick_end": int(self.round_index * self.config.decision_interval + window_ticks - 1),
            "decision_interval_ticks": int(window_ticks),
            "reward": float(reward),
            "chr": float(metrics.chr),
            "local_rsu_chr": float(metrics.local_rsu_chr),
            "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
            "latency": latency,
            "request_count": int(metrics.request_count),
            "evaluation_request_source": "current_service_window_true_requests",
            "coverage_source": "service_window_swept_coverage",
            "decision_signal_source": "mobility_and_training_history_only",
            "uses_predicted_content_demands": False,
            "uses_vehicle_level_miss_feedback": False,
            "uses_content_level_feedback": False,
            "selected_hotspot": _sanitize_hotspot(selected_hotspot),
            "action": decoded.to_dict(),
            "lambda_smooth": float(lambda_smooth),
            "lambda_smooth_source": "system_auto_rule",
            "path_plan_status": path_plan.status,
            "path_plan_solver": path_plan.solver,
            "mrsu_position": float(self.env.mrsu.position),
            "mrsu_speed": float(self.env.mrsu.speed),
            "mrsu_cache": [int(x) for x in mrsu_cache],
            "frsu_cache": [int(x) for x in frsu_cache],
            "mrsu_covered": [int(x) for x in coverage.mrsu_covered],
            "frsu_covered": [int(x) for x in coverage.frsu_covered],
            "decision_mrsu_covered": [int(x) for x in decision_coverage.mrsu_covered],
            "decision_frsu_covered": [int(x) for x in decision_coverage.frsu_covered],
            "overlap": [int(x) for x in coverage.overlap],
            "metrics": metrics.to_dict(),
        }

        self.round_index += 1
        done = self.round_index >= self.rounds
        if not done:
            self._prepare_round()
        else:
            self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        return self.current_state.copy(), float(reward), bool(done), info

    def _prepare_round(self) -> None:
        if self.env is None or self.hotspot_generator is None:
            raise RuntimeError("Scenario must be reset before preparing a round.")
        generated = self.hotspot_generator.generate(
            self.env.mobility.positions(),
            self.vehicle_history_demands,
        )
        self.current_hotspots = [hotspot.to_dict() for hotspot in generated]
        self.current_state = self._build_state()

    def _build_state(self) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Scenario must be reset before building state.")
        coverage = self.env.coverage_snapshot()
        vehicle_num = max(int(self.config.vehicle_num), 1)
        road_length = max(float(self.config.road_length), 1e-9)
        speed_scale = max(float(self.config.mrsu_v_max), float(self.config.max_vehicle_speed), 1.0)
        last = self.env.last_metrics
        request_count = max(float(last.request_count), 1.0)
        global_features = [
            float(self.round_index) / max(float(self.rounds), 1.0),
            (float(self.env.mrsu.position) % road_length) / road_length,
            float(self.env.mrsu.speed) / speed_scale,
            (float(self.env.frsu.position) % road_length) / road_length,
            len(coverage.mrsu_covered) / float(vehicle_num),
            len(coverage.frsu_covered) / float(vehicle_num),
            len(coverage.overlap) / float(vehicle_num),
            float(last.chr),
            float(last.not_covered_count) / request_count,
            float(last.not_cached_count) / request_count,
            float(last.mrsu_hit_count) / request_count,
            float(last.frsu_hit_count) / request_count,
            float(self.config.mrsu_cache_capacity) / max(float(self.config.movie_num), 1.0),
        ]

        strengths = [
            self._history_strength(hotspot.get("covered_vehicle_ids", []))
            for hotspot in self.current_hotspots
        ]
        max_strength = max(max(strengths, default=0.0), 1.0)
        candidate_features: List[float] = []
        for idx in range(self.candidate_count):
            if idx >= len(self.current_hotspots):
                candidate_features.extend([0.0] * CANDIDATE_FEATURE_DIM)
                continue
            hotspot = self.current_hotspots[idx]
            position = float(hotspot.get("position", 0.0)) % road_length
            forward_distance = CoverageModel.forward_distance(self.env.mrsu.position, position, road_length)
            circular_distance = CoverageModel.circular_distance(self.env.mrsu.position, position, road_length)
            covered_ids = [int(x) for x in hotspot.get("covered_vehicle_ids", [])]
            avg_speed = self._average_vehicle_speed(covered_ids)
            candidate_features.extend(
                [
                    position / road_length,
                    forward_distance / road_length,
                    circular_distance / road_length,
                    len(covered_ids) / float(vehicle_num),
                    self._history_strength(covered_ids) / max_strength,
                    avg_speed / speed_scale,
                ]
            )
        state = np.array(global_features + candidate_features, dtype=np.float32)
        if len(state) != self.state_dim:
            raise RuntimeError(f"Unexpected DQN state dimension: {len(state)} != {self.state_dim}")
        return state

    def _average_vehicle_speed(self, vehicle_ids: Sequence[int]) -> float:
        if self.env is None or not vehicle_ids:
            return 0.0
        speed_by_id = {
            int(vehicle.vehicle_id): float(vehicle.speed)
            for vehicle in self.env.vehicle_states()
        }
        values = [speed_by_id.get(int(vehicle_id), 0.0) for vehicle_id in vehicle_ids]
        return float(sum(values) / max(len(values), 1))

    def _select_hotspot(self, hotspot_index: int) -> dict:
        if not self.current_hotspots:
            return _fallback_hotspot(float(self.env.mrsu.position if self.env else 0.0))
        index = min(max(int(hotspot_index), 0), len(self.current_hotspots) - 1)
        return self.current_hotspots[index]

    def _build_caches(
        self,
        decoded: DecodedAction,
        coverage,
        selected_hotspot: dict,
    ) -> Tuple[List[int], List[int]]:
        if self.env is None:
            raise RuntimeError("Scenario must be reset before building caches.")
        mrsu_counter = Counter()
        frsu_counter = Counter()

        mrsu_ids = [int(x) for x in coverage.mrsu_covered]
        frsu_ids = [int(x) for x in coverage.frsu_covered]
        hotspot_ids = [int(x) for x in selected_hotspot.get("covered_vehicle_ids", [])]
        frsu_only_ids = sorted(set(frsu_ids) - set(mrsu_ids))

        if decoded.mrsu_cache_policy == "mrsu_current_history":
            mrsu_counter = self._history_counter(mrsu_ids)
        elif decoded.mrsu_cache_policy == "selected_hotspot_history":
            mrsu_counter = self._history_counter(hotspot_ids)
        elif decoded.mrsu_cache_policy == "mrsu_current_plus_hotspot_history":
            mrsu_counter.update(self._history_counter(mrsu_ids))
            mrsu_counter.update(self._history_counter(hotspot_ids))

        if decoded.frsu_cache_policy == "frsu_current_history":
            frsu_counter = self._history_counter(frsu_ids)
        elif decoded.frsu_cache_policy == "frsu_only_history":
            frsu_counter = self._history_counter(frsu_only_ids)
        elif decoded.frsu_cache_policy == "frsu_current_plus_frsu_only_history":
            frsu_counter.update(self._history_counter(frsu_ids))
            frsu_counter.update(self._history_counter(frsu_only_ids))

        mrsu_priority = (
            self.env.global_top_contents
            if decoded.mrsu_cache_policy == "global_history"
            else [content_id for content_id, _ in mrsu_counter.most_common()]
        )
        frsu_priority = (
            self.env.global_top_contents
            if decoded.frsu_cache_policy == "global_history"
            else [content_id for content_id, _ in frsu_counter.most_common()]
        )
        mrsu_cache = _fill_unique(mrsu_priority, self.config.mrsu_cache_capacity, self.env.global_top_contents)
        frsu_cache = _fill_unique(frsu_priority, self.config.frsu_cache_capacity, self.env.global_top_contents)
        return mrsu_cache, frsu_cache

    def _history_counter(self, vehicle_ids: Iterable[int]) -> Counter:
        counter = Counter()
        for vehicle_id in vehicle_ids:
            counter.update(self.vehicle_history_demands.get(int(vehicle_id), Counter()))
        return counter

    def _history_strength(self, vehicle_ids: Iterable[int]) -> float:
        return float(sum(self._history_counter(vehicle_ids).values()))

    def _reward(self, metrics: RoundMetrics) -> float:
        request_count = max(float(metrics.request_count), 1.0)
        reward = float(metrics.chr)
        reward -= self.reward_not_covered_penalty * float(metrics.not_covered_count) / request_count
        reward -= self.reward_not_cached_penalty * float(metrics.not_cached_count) / request_count
        return float(reward)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=int(capacity))
        self.rng = random.Random(int(seed))

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.98,
        batch_size: int = 64,
        buffer_size: int = 20000,
        target_update_steps: int = 200,
        device: str = "auto",
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for DQN. Install torch or use the torchGPU environment.")
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_steps = int(target_update_steps)
        self.learn_steps = 0
        self.rng = random.Random(int(seed))

        self.policy_net = QNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self.replay = ReplayBuffer(buffer_size, seed)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < float(epsilon):
            return self.rng.randrange(self.output_dim)
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def observe(self, state, action: int, reward: float, next_state, done: bool) -> float | None:
        self.replay.push(state, action, reward, next_state, done)
        if len(self.replay) < self.batch_size:
            return None
        return self.learn()

    def learn(self) -> float:
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, output_path: str | Path, metadata: dict) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "metadata": metadata,
            },
            str(output_path),
        )

    def load_state(self, state_dict: dict) -> None:
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()


def train_dqn(
    config: MRSUSimulationConfig,
    output_dir: str,
    episodes: int = 200,
    rounds: int | None = None,
    seed: int = 42,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.98,
    batch_size: int = 64,
    buffer_size: int = 20000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.985,
    target_update_steps: int = 200,
    device: str = "auto",
    reward_not_covered_penalty: float = 0.0,
    reward_not_cached_penalty: float = 0.0,
    log_csv_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    scenario = DQNBaselineScenario(
        config=config,
        rounds=rounds if rounds is not None else config.rounds,
        seed_offset=0,
        candidate_count=config.candidate_count,
        reward_not_covered_penalty=reward_not_covered_penalty,
        reward_not_cached_penalty=reward_not_cached_penalty,
        latency_model=CV2XLatencyModel(),
    )
    agent = DQNAgent(
        input_dim=scenario.state_dim,
        output_dim=scenario.action_dim,
        seed=seed,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_steps=target_update_steps,
        device=device,
    )
    epsilon = float(epsilon_start)
    logs: List[dict] = []
    if log_csv_path:
        _init_training_log_csv(log_csv_path)

    for episode in range(int(episodes)):
        scenario = DQNBaselineScenario(
            config=config,
            rounds=rounds if rounds is not None else config.rounds,
            seed_offset=episode,
            candidate_count=config.candidate_count,
            reward_not_covered_penalty=reward_not_covered_penalty,
            reward_not_cached_penalty=reward_not_cached_penalty,
            latency_model=CV2XLatencyModel(),
        )
        state = scenario.reset()
        done = False
        episode_reward = 0.0
        round_chr: List[float] = []
        round_local_rsu_chr: List[float] = []
        round_delay_ms: List[float] = []
        losses: List[float] = []
        while not done:
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, info = scenario.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(float(loss))
            state = next_state
            episode_reward += float(reward)
            round_chr.append(float(info["chr"]))
            round_local_rsu_chr.append(float(info.get("local_rsu_chr", info["chr"])))
            round_delay_ms.append(float(info.get("round_delay_ms", 0.0)))

        row = {
            "episode": int(episode),
            "seed": int(config.seed) + int(episode),
            "epsilon": float(epsilon),
            "episode_reward": float(episode_reward),
            "episode_average_chr": _mean(round_chr),
            "episode_average_local_rsu_chr": _mean(round_local_rsu_chr),
            "episode_average_delay_ms": _mean(round_delay_ms),
            "loss": _mean(losses),
            "rounds": len(round_chr),
            "mrsu_cache_capacity": int(config.mrsu_cache_capacity),
            "frsu_cache_capacity": int(config.frsu_cache_capacity),
        }
        logs.append(row)
        if log_csv_path:
            _append_training_log_csv(log_csv_path, row)
        if verbose:
            print(
                f"[DQN train] episode={episode:04d} "
                f"reward={episode_reward:.4f} avg_chr={row['episode_average_chr']:.4f} "
                f"avg_delay={row['episode_average_delay_ms']:.2f}ms "
                f"epsilon={epsilon:.4f} loss={row['loss']:.6f}"
            )
        epsilon = max(float(epsilon_end), float(epsilon) * float(epsilon_decay))

    metadata = dqn_metadata(config, scenario.state_dim, scenario.action_dim, agent.device.type, hidden_dim)
    model_path = Path(output_dir) / f"dqn_model_capacity_{int(config.mrsu_cache_capacity)}.pt"
    agent.save(model_path, metadata)
    return {
        "model_path": str(model_path),
        "metadata": metadata,
        "training_log": logs,
    }


def evaluate_dqn(
    config: MRSUSimulationConfig,
    model_path: str | Path,
    seed: int | None = None,
    device: str = "auto",
    latency_model: CV2XLatencyModel | None = None,
    verbose: bool = True,
) -> dict:
    if torch is None:
        raise RuntimeError("PyTorch is required for DQN evaluation.")
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    candidate_count = int(metadata.get("candidate_count", config.candidate_count))
    eval_config = replace(config, seed=int(config.seed if seed is None else seed))
    scenario = DQNBaselineScenario(
        config=eval_config,
        rounds=eval_config.rounds,
        seed_offset=0,
        candidate_count=candidate_count,
        latency_model=latency_model or CV2XLatencyModel(),
    )
    input_dim = int(metadata.get("state_dim", scenario.state_dim))
    output_dim = int(metadata.get("action_dim", scenario.action_dim))
    agent = DQNAgent(
        input_dim=input_dim,
        output_dim=output_dim,
        seed=int(eval_config.seed) + 811,
        hidden_dim=int(metadata.get("hidden_dim", 128)),
        device=device,
    )
    agent.load_state(checkpoint["model_state_dict"])

    state = scenario.reset()
    done = False
    round_metrics: List[RoundMetrics] = []
    round_logs: List[dict] = []
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, info = scenario.step(action)
        state = next_state
        metrics = RoundMetrics(**info["metrics"])
        round_metrics.append(metrics)
        round_logs.append(
            {
                "round": int(info["round"]),
                "physical_tick_start": int(info.get("physical_tick_start", 0)),
                "physical_tick_end": int(info.get("physical_tick_end", 0)),
                "decision_interval_ticks": int(info.get("decision_interval_ticks", eval_config.decision_interval)),
                "chr": float(info["chr"]),
                "round_delay_ms": float(info.get("round_delay_ms", 0.0)),
                "latency": info.get("latency", {}),
                "reward": float(reward),
                "request_count": int(info["request_count"]),
                "decision_request_count": "",
                "evaluation_request_count": int(info["request_count"]),
                "decision_request_source": "mobility_and_training_history_only",
                "uses_proposed_miss_feedback": False,
                "uses_content_level_feedback": False,
                "uses_predicted_content_demands": False,
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "selected_hotspot": info["selected_hotspot"],
                "lambda_smooth": float(info["lambda_smooth"]),
                "lambda_smooth_source": "system_auto_rule",
                "path_plan_status": info["path_plan_status"],
                "path_plan_solver": info["path_plan_solver"],
                "mrsu_position": float(info["mrsu_position"]),
                "mrsu_speed": float(info["mrsu_speed"]),
                "mrsu_cache": info["mrsu_cache"],
                "frsu_cache": info["frsu_cache"],
                "mrsu_covered": info["mrsu_covered"],
                "frsu_covered": info["frsu_covered"],
                "decision_mrsu_covered": info.get("decision_mrsu_covered", []),
                "decision_frsu_covered": info.get("decision_frsu_covered", []),
                "overlap": info["overlap"],
                "metrics": info["metrics"],
                "decision_details": {
                    "policy": "dqn_discrete_history_policy",
                    "action": info["action"],
                    "model_path": str(model_path),
                    "state_features_exclude_content_ids": True,
                    "cache_source": "training_history_only",
                },
            }
        )
        if verbose:
            print(
                f"[DQN] round={info['round']:02d} LocalCHR={metrics.local_rsu_chr:.4f} "
                f"mRSU_hit={metrics.mrsu_hit_count} fRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count} "
                f"Delay={float(info.get('round_delay_ms', 0.0)):.2f}ms"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(summarize_round_latencies(log.get("latency") for log in round_logs))
    summary.update(
        {
            "method": METHOD_DQN,
            "method_label": METHOD_LABEL,
            "method_note": METHOD_NOTE,
            "decision_request_source": "mobility_and_training_history_only",
            "uses_proposed_miss_feedback": False,
            "uses_content_level_feedback": False,
            "uses_predicted_content_demands": False,
            "physical_rounds": int(eval_config.rounds),
            "decision_interval_ticks": int(eval_config.decision_interval),
            "decision_rounds": int(eval_config.decision_rounds),
            "latency_model": (latency_model or scenario.latency_model).config.to_dict(),
            "reward_includes_delay": False,
            "round_chr": [float(item.chr) for item in round_metrics],
            "round_local_rsu_chr": [
                local_rsu_chr_from_counts(item.hit_count, item.not_cached_count)
                for item in round_metrics
            ],
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def dqn_metadata(
    config: MRSUSimulationConfig,
    input_dim: int,
    output_dim: int,
    device: str,
    hidden_dim: int = 128,
) -> dict:
    return {
        "method": METHOD_DQN,
        "method_label": METHOD_LABEL,
        "method_note": METHOD_NOTE,
        "state_dim": int(input_dim),
        "action_dim": int(output_dim),
        "candidate_count": int(config.candidate_count),
        "physical_rounds": int(config.rounds),
        "decision_interval_ticks": int(config.decision_interval),
        "decision_rounds": int(config.decision_rounds),
        "hidden_dim": int(hidden_dim),
        "mrsu_cache_policies": list(MRSU_CACHE_POLICIES),
        "frsu_cache_policies": list(FRSU_CACHE_POLICIES),
        "state_excludes": [
            "movie_ids",
            "predicted_request_content_lists",
            "demand_summary",
            "dominant_contents",
            "vehicle_level_miss_content_feedback",
        ],
        "cache_content_source": "MovieLens training history profiles and global training popularity",
        "reward_includes_delay": False,
        "device": str(device),
        "config": asdict(config),
    }


def _fill_unique(priority_items: Iterable[int], capacity: int, fallback_items: Iterable[int]) -> List[int]:
    result: List[int] = []
    seen = set()
    for item in list(priority_items) + list(fallback_items):
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
        if len(result) >= int(capacity):
            break
    return result


def _sanitize_hotspot(hotspot: dict) -> dict:
    return {
        "hotspot_id": int(hotspot.get("hotspot_id", 0)),
        "position": float(hotspot.get("position", 0.0)),
        "covered_vehicle_ids": [int(x) for x in hotspot.get("covered_vehicle_ids", [])],
        "covered_vehicle_count": int(hotspot.get("covered_vehicle_count", 0)),
        "historical_cache_gain": float(hotspot.get("potential_cache_gain", 0.0)),
    }


def _fallback_hotspot(position: float = 0.0) -> dict:
    return {
        "hotspot_id": 0,
        "position": float(position),
        "covered_vehicle_ids": [],
        "covered_vehicle_count": 0,
        "potential_cache_gain": 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _init_training_log_csv(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_training_log_fields())
        writer.writeheader()


def _append_training_log_csv(path: str | Path, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_training_log_fields())
        writer.writerow({field: row.get(field, "") for field in _training_log_fields()})


def _training_log_fields() -> List[str]:
    return [
        "episode",
        "seed",
        "epsilon",
        "episode_reward",
        "episode_average_chr",
        "episode_average_local_rsu_chr",
        "episode_average_delay_ms",
        "loss",
        "rounds",
        "mrsu_cache_capacity",
        "frsu_cache_capacity",
    ]
