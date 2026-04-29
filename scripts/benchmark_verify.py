#!/usr/bin/env python3
"""Benchmark verify latency with/without ark thinking=disabled.

Replays saved verify debug artifacts against the real ark endpoint.
Usage:
    python scripts/benchmark_verify.py [session_dir] [--rounds N]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / ".tmp_migrated_vla_config.json"


def load_ark_config() -> tuple[str, str, str]:
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    ark = cfg["arm"]["verify"]["providers"]["ark"]
    return ark["api_base"], ark["api_key"], ark["model"]


def build_payload(
    session_dir: Path,
    model: str,
    request_meta: dict,
    system_prompt_override: str | None = None,
    max_tokens_override: int | None = None,
) -> dict:
    """Rebuild production-format payload from saved debug artifacts."""
    system_prompt = system_prompt_override or request_meta["messages"][0]["content"]
    instruction = request_meta["instruction"]
    completion_mode = request_meta["completion_mode"]
    images = request_meta["images"]

    def b64(name: str) -> str:
        return base64.b64encode((session_dir / images[name]).read_bytes()).decode()

    user_content = [
        {"type": "text", "text": f"Instruction: {instruction}\nCompletion mode: {completion_mode}"},
        {"type": "text", "text": "[baseline_main] Desk-wide view before the task."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64('baseline_main')}"}},
        {"type": "text", "text": "[baseline_wrist] Close-up wrist view before the task."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64('baseline_wrist')}"}},
        {"type": "text", "text": "[current_main] Desk-wide view now."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64('current_main')}"}},
        {"type": "text", "text": "[current_wrist] Close-up wrist view now."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64('current_wrist')}"}},
    ]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens_override if max_tokens_override is not None else request_meta.get("max_tokens", 256),
        "temperature": 0,
    }


async def run_one(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> tuple[float, str, str]:
    t0 = time.perf_counter()
    resp = await client.post(url, headers=headers, json=payload)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return elapsed, content, data.get("id", "")


async def benchmark(
    session_dir: Path,
    rounds: int,
    model_override: str | None = None,
    prompt_variants: list[tuple[str, Path | None]] | None = None,
) -> None:
    api_base, api_key, model = load_ark_config()
    if model_override:
        model = model_override
    request_files = sorted(session_dir.glob("*-request.json"))
    if not request_files:
        raise FileNotFoundError(f"No request.json under {session_dir}")
    request_meta = json.loads(request_files[0].read_text())

    variants: list[tuple[str, dict]] = []
    if prompt_variants:
        for label, prompt_path in prompt_variants:
            sys_prompt = prompt_path.read_text() if prompt_path else None
            max_toks = 8 if prompt_path and "minimal" in prompt_path.name else None
            payload = build_payload(session_dir, model, request_meta, sys_prompt, max_toks)
            payload["thinking"] = {"type": "disabled"}
            variants.append((label, payload))
    else:
        payload_base = build_payload(session_dir, model, request_meta)
        payload_nothink = {**payload_base, "thinking": {"type": "disabled"}}
        variants = [
            ("thinking=ON (default)  ", payload_base),
            ("thinking=DISABLED      ", payload_nothink),
        ]

    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print(f"Session: {session_dir.name}")
    print(f"Model:   {model}")
    print(f"Rounds:  {rounds} per variant\n")

    async with httpx.AsyncClient(timeout=120.0, trust_env=True, http2=True) as client:
        results = {}
        for label, payload in variants:
            times: list[float] = []
            for i in range(rounds):
                elapsed, content, req_id = await run_one(client, url, headers, payload)
                times.append(elapsed)
                snippet = content.replace("\n", " ")[:100]
                print(f"  [{label.strip()}] round {i+1}: {elapsed*1000:8.0f} ms  id=…{req_id[-10:]}  {snippet}")
            avg = statistics.mean(times) * 1000
            mn, mx = min(times) * 1000, max(times) * 1000
            results[label] = (avg, mn, mx)
            print(f"  >>> {label}: avg={avg:7.0f} ms  min={mn:7.0f} ms  max={mx:7.0f} ms\n")

        labels = list(results.keys())
        if len(labels) >= 2:
            base_avg = results[labels[0]][0]
            for lbl in labels[1:]:
                cur_avg = results[lbl][0]
                saved = base_avg - cur_avg
                pct = saved / base_avg * 100 if base_avg else 0
                print(f"{lbl.strip()} vs {labels[0].strip()}: {saved:+.0f} ms  ({pct:+.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", nargs="?")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default=None, help="Override ark model id")
    parser.add_argument(
        "--compare-prompts",
        action="store_true",
        help="Compare full verify.md vs verify_minimal.md (both with thinking=disabled)",
    )
    args = parser.parse_args()

    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        sessions = sorted((ROOT / "debug" / "verify").glob("rev*-request.json"))
        session_dir = sessions[-1].parent

    prompt_variants = None
    if args.compare_prompts:
        mission_dir = ROOT / "missions" / "desk_cleanup"
        prompt_variants = [
            ("full prompt (JSON out)  ", mission_dir / "verify.md"),
            ("minimal prompt (1-word)  ", mission_dir / "verify_minimal.md"),
        ]

    asyncio.run(benchmark(session_dir, args.rounds, args.model, prompt_variants))


if __name__ == "__main__":
    main()
