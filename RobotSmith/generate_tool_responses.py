#!/usr/bin/env python3
"""Generate RobotSmith tool design JSON responses with GPT-5.4."""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock

project_path = Path(__file__).resolve().parent


def parse_json_response(response: str):
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    return json.loads(response)


def safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "tool"


def fallback_tool_name(description: str) -> str:
    return description.split(":", 1)[0].strip() if ":" in description else description.strip()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10000):
        candidate = path.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique filename for: {path}")


def call_model(client, model: str, prompt: str, effort: str) -> str:
    chunks = []
    with client.responses.stream(
        model=model,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": effort},
    ) as stream:
        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                chunks.append(event.delta)

        stream.get_final_response()

    return "".join(chunks)


def setup_run_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"generate_tool_responses_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("generate_tool_responses")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger, log_path


def main():
    parser = argparse.ArgumentParser(description="Generate tool design JSON responses from descriptions.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=project_path / "tmp_response.txt",
        help="Text file containing one tool description per non-empty line.",
    )
    parser.add_argument(
        "--template-file",
        type=Path,
        default=project_path / "utils" / "template_tool_design_manual.txt",
        help="Prompt template containing $GOAL_DESCRIPTION$.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_path / "responses",
        help="Directory to write generated tool JSON files.",
    )
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--base-url", type=str, default="http://43.106.115.130:8080")
    parser.add_argument("--effort", type=str, default="medium", choices=["low", "medium", "high", "xhigh"], help="GPT reasoning effort.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of lines to process.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    parser.add_argument("--workers", type=int, default=1, help="Maximum number of concurrent model requests.")
    parser.add_argument("--print-io", action="store_true", help="Print each prompt and model response.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without sending model requests.")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    template = args.template_file.read_text()
    descriptions = [line.strip() for line in args.input_file.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        descriptions = descriptions[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_run_logger(args.output_dir)
    logger.info("run started")
    logger.info(
        "args input_file=%s template_file=%s output_dir=%s model=%s base_url=%s effort=%s "
        "limit=%s sleep=%s workers=%s print_io=%s dry_run=%s",
        args.input_file,
        args.template_file,
        args.output_dir,
        args.model,
        args.base_url,
        args.effort,
        args.limit,
        args.sleep,
        args.workers,
        args.print_io,
        args.dry_run,
    )

    client_factory = None
    if not args.dry_run:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        import httpx
        from openai import OpenAI

        def client_factory():
            return OpenAI(
                api_key=api_key,
                base_url=args.base_url,
                http_client=httpx.Client(trust_env=False),
            )

    print(f"Loaded {len(descriptions)} non-empty description(s).")
    print(f"Writing responses to: {args.output_dir}")
    print(f"Run log: {log_path}")
    logger.info("loaded %d non-empty descriptions", len(descriptions))

    output_lock = Lock()
    print_lock = Lock()

    def process_description(idx: int, description: str):
        prompt = template.replace("$GOAL_DESCRIPTION$", description)
        logger.info("[%d/%d] start: %s", idx, len(descriptions), description)

        if args.dry_run:
            logger.info("[%d/%d] dry-run skipped request", idx, len(descriptions))
            return {
                "idx": idx,
                "description": description,
                "prompt": prompt,
                "response_text": None,
                "output_path": None,
                "parse_error": None,
                "request_error": None,
                "dry_run": True,
            }

        try:
            client = client_factory()
            response_text = call_model(client, args.model, prompt, args.effort)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%d/%d] request failed: %s", idx, len(descriptions), description)
            return {
                "idx": idx,
                "description": description,
                "prompt": prompt,
                "response_text": None,
                "output_path": None,
                "parse_error": None,
                "request_error": str(exc),
                "dry_run": False,
            }

        logger.info("[%d/%d] response chars=%d", idx, len(descriptions), len(response_text))

        fallback_name = fallback_tool_name(description)
        parse_error = None
        try:
            design_json = parse_json_response(response_text)
            tool_name = design_json.get("name") or fallback_name
            with output_lock:
                output_path = unique_path(args.output_dir / f"{safe_filename(tool_name)}.json")
                output_path.write_text(json.dumps(design_json, indent=2))
            logger.info("[%d/%d] saved parsed JSON: %s", idx, len(descriptions), output_path)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
            with output_lock:
                output_path = unique_path(args.output_dir / f"{safe_filename(fallback_name)}.json")
                output_path.write_text(
                    json.dumps(
                        {
                            "tool_description": description,
                            "parse_error": parse_error,
                            "raw_response": response_text,
                        },
                        indent=2,
                    )
                )
            logger.exception("[%d/%d] failed to parse/save model response: %s", idx, len(descriptions), description)

        return {
            "idx": idx,
            "description": description,
            "prompt": prompt,
            "response_text": response_text,
            "output_path": output_path,
            "parse_error": parse_error,
            "request_error": None,
            "dry_run": False,
        }

    def report_result(result):
        with print_lock:
            idx = result["idx"]
            description = result["description"]
            print(f"\n[{idx}/{len(descriptions)}] {description}")
            if args.print_io or args.dry_run:
                print("\n----- PROMPT START -----")
                print(result["prompt"])
                print("----- PROMPT END -----")

            if result["dry_run"]:
                print("  dry-run: skipped model request")
                return

            if result["request_error"]:
                print(f"  ERROR: request failed: {result['request_error']}")
                return

            if args.print_io:
                print("\n----- MODEL RESPONSE START -----")
                print(result["response_text"] or "(empty response)")
                print("----- MODEL RESPONSE END -----")

            if result["parse_error"]:
                print(f"  WARNING: failed to parse model response as JSON: {result['parse_error']}")
                print(f"  saved raw response wrapper: {result['output_path']}")
            else:
                print(f"  saved parsed JSON: {result['output_path']}")

    if args.workers == 1:
        for idx, description in enumerate(descriptions, start=1):
            report_result(process_description(idx, description))
            if args.sleep > 0 and idx < len(descriptions):
                time.sleep(args.sleep)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for idx, description in enumerate(descriptions, start=1):
                futures.append(executor.submit(process_description, idx, description))
                if args.sleep > 0 and idx < len(descriptions):
                    time.sleep(args.sleep)

            for future in concurrent.futures.as_completed(futures):
                report_result(future.result())

    logger.info("run finished")


if __name__ == "__main__":
    main()
