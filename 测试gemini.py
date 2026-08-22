import argparse
import json
import getpass
import requests


def main():
    parser = argparse.ArgumentParser(description="Test gptsapi Gemini generateContent API")
    parser.add_argument("--api-key", default=None, help="Your API key")
    parser.add_argument("--model", default="gemini-3.5-flash", help="Gemini model name")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。", help="User prompt")
    parser.add_argument("--base-url", default="https://api.gptsapi.net", help="Base URL")
    args = parser.parse_args()

    api_key = args.api_key or getpass.getpass("请输入 API key: ")

    url = f"{args.base_url.rstrip('/')}/v1beta/models/{args.model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": args.prompt
                    }
                ]
            }
        ]
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
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
        print(text)
    except Exception:
        print("返回结构解析失败，完整响应如下：")
        print(json.dumps(data, ensure_ascii=False, indent=2))

    if "usageMetadata" in data:
        print("-" * 60)
        print("Token 使用情况：")
        print(json.dumps(data["usageMetadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()