#!/usr/bin/env python3
"""
Test whether an API key can send a simple message to GPT-5.4.

Examples:
  OPENAI_API_KEY=sk-... python test_gpt54_api.py
  python test_gpt54_api.py --api-key sk-... --base-url http://43.106.115.130:8080
"""

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a minimal Responses API request to GPT-5.4 to validate an API key."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://43.106.115.130:8080"),
        help="Optional API base URL. Defaults to OPENAI_BASE_URL or http://43.106.115.130:8080.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Model name to test. Default: gpt-5.4.",
    )
    parser.add_argument(
        "--message",
        default="请用一句话回复：GPT-5.4 API 连通性测试成功。",
        help="Test message to send.",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh"],
        help="GPT reasoning effort. Default: medium.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.api_key:
        print("ERROR: missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        import httpx
        from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(f"Missing dependency: {missing}. Install with: pip install openai httpx", file=sys.stderr)
        return 2

    client_kwargs = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "http_client": httpx.Client(trust_env=False),
    }

    client = OpenAI(**client_kwargs)

    try:
        chunks = []
        with client.responses.stream(
            model=args.model,
            input=[{"role": "user", "content": args.message}],
            reasoning={"effort": args.effort},
        ) as stream:
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    chunks.append(event.delta)
                    print(event.delta, end="", flush=True)

            final_response = stream.get_final_response()
    except APIStatusError as exc:
        print(f"FAILED: API returned HTTP {exc.status_code}", file=sys.stderr)
        print(exc.response.text, file=sys.stderr)
        return 1
    except APIConnectionError as exc:
        print(f"FAILED: connection error: {exc}", file=sys.stderr)
        return 1
    except OpenAIError as exc:
        print(f"FAILED: OpenAI SDK error: {exc}", file=sys.stderr)
        return 1

    output_text = "".join(chunks)
    if not output_text:
        print("FAILED: request completed but no output_text delta was received.", file=sys.stderr)
        print("raw_response:")
        print(final_response.model_dump_json(indent=2))
        return 1

    print("\nSUCCESS: GPT-5.4 API request completed.")
    print(f"model: {getattr(final_response, 'model', args.model)}")
    if getattr(final_response, "usage", None):
        print(f"usage: {final_response.usage}")
    print("response:")
    print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
