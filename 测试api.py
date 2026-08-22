import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(description="Test gptsapi OpenAI-compatible API")
    parser.add_argument("--api-key", required=True, help="Your API key")
    parser.add_argument("--base-url", default="https://api.gptsapi.net/v1", help="API base url")
    parser.add_argument("--model", default="grok-4.3", help="Model name")
    parser.add_argument("--prompt", default="你好，请简单介绍一下你自己。", help="User prompt")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max output tokens")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        "max_tokens": args.max_tokens,
    }

    print("=" * 60)
    print("Request URL:", url)
    print("Model:", args.model)
    print("Prompt:", args.prompt)
    print("=" * 60)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print("请求失败：")
        print(e)
        return

    print("HTTP Status:", response.status_code)

    try:
        data = response.json()
    except Exception:
        print("返回内容不是 JSON：")
        print(response.text)
        return

    if response.status_code != 200:
        print("接口返回错误：")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print("接口调用成功。")
    print("-" * 60)

    try:
        content = data["choices"][0]["message"]["content"]
        print(content)
    except KeyError:
        print("返回结构不符合 Chat Completions 标准，完整响应如下：")
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()