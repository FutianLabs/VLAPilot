#!/usr/bin/env python3
"""Single-arm VLA Agent entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-arm VLA Agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file path (default: ~/.vlapilot/config.json, fallback: ~/.vla/config.json)",
    )
    parser.add_argument(
        "--instruction",
        "-i",
        required=True,
        help="Mission instruction passed to the planner",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save planner LLM calls and verifier artifacts to disk",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    from vlapilot.agent import VLAAgent

    agent = VLAAgent(config_path=args.config, debug=args.debug)
    try:
        result = await agent.run(args.instruction)
    finally:
        await agent.cleanup()

    print(f"status: {result.get('status')}")
    if result.get("plan_id"):
        print(f"plan_id: {result['plan_id']}")
    if result.get("results"):
        print(f"steps: {len(result['results'])}")
    if result.get("error"):
        print(f"error: {result['error']}")
    return result


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
