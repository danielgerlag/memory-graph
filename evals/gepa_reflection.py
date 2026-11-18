#!/usr/bin/env python3
"""Call the configured model once for GEPA reflection in an isolated host environment."""

import asyncio
import json
import os
import sys
from pathlib import Path

import run_behavior
import run_inspect
from controlled_core import validate_options
from profiles import apply_profiles


async def generate(args, prompt):
    from inspect_ai.model import GenerateConfig, get_model

    run_inspect.pricing_config(args)
    model = get_model(args.model, memoize=False, **run_inspect.model_args(args))
    result = await model.generate(
        prompt,
        config=GenerateConfig(max_tokens=args.max_tokens, timeout=args.time_limit, max_retries=0),
    )
    if not result.choices or not result.completion or result.choices[0].stop_reason != "stop":
        raise ValueError("reflection completion was missing or truncated")
    usage = result.usage
    return {
        "completion": result.completion,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
    }


def main():
    args = run_behavior.parser().parse_args()
    try:
        apply_profiles(args, runner="controlled")
        validate_options(args, os.environ)
        prompt = sys.stdin.read(65537)
        if not prompt or len(prompt.encode("utf-8")) > 65536:
            raise ValueError("reflection prompt is empty or too large")
        root = Path(args.log_dir).resolve()
        if not root.is_relative_to(Path.cwd().resolve()):
            raise ValueError("reflection log directory must be inside the current directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        run_inspect.isolate_environment(args, root)
        with run_inspect.azure_credential_log_guard(args):
            run_inspect.preflight_default_credentials(args)
            response = asyncio.run(generate(args, prompt))
        print(json.dumps(response))
        return 0
    except Exception as exc:  # noqa: BLE001 - prevent accidental credential disclosure
        print(f"reflection failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
