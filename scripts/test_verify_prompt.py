#!/usr/bin/env python3
"""Test verify prompt against the test set."""

import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TEST_SET_DIR = ROOT / "verify_test_set"


def load_config():
    """Load verify config from ~/.vlapilot/config.json, falling back to ~/.vla/config.json."""
    candidates = [Path.home() / ".vlapilot" / "config.json", Path.home() / ".vla" / "config.json"]
    config_path = next((path for path in candidates if path.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"Config not found. Checked: {', '.join(str(p) for p in candidates)}")
    cfg = json.loads(config_path.read_text())
    verify_cfg = cfg["arm"]["verify"]
    provider = verify_cfg["provider"]
    provider_cfg = verify_cfg["providers"][provider]
    return provider, provider_cfg


async def call_verify_api(
    provider: str,
    provider_cfg: dict,
    system_prompt: str,
    instruction: str,
    completion_mode: str,
    images: dict[str, bytes],
) -> str:
    """Call the verify API and return the verdict."""
    # Build message content
    user_content = [
        {"type": "text", "text": f"Instruction: {instruction}\nCompletion mode: {completion_mode}"},
        {"type": "text", "text": "[baseline_main] Desk-wide view before the task."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(images['baseline_main']).decode()}"}},
        {"type": "text", "text": "[baseline_wrist] Close-up wrist view before the task."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(images['baseline_wrist']).decode()}"}},
        {"type": "text", "text": "[current_main] Desk-wide view now."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(images['current_main']).decode()}"}},
        {"type": "text", "text": "[current_wrist] Close-up wrist view now."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(images['current_wrist']).decode()}"}},
    ]

    payload = {
        "model": provider_cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 256,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }

    url = f"{provider_cfg['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider_cfg['api_key']}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0, trust_env=True, http2=True) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content


async def test_case_by_prefix(
    provider: str,
    provider_cfg: dict,
    system_prompt: str,
    test_dir: Path,
    case_prefix: str,
    expected_verdict: str,
) -> tuple[bool, str, str]:
    """Test a single case by prefix. Returns (passed, actual_verdict, case_name)."""
    # Load request
    request_file = test_dir / f"{case_prefix}-request.json"
    if not request_file.exists():
        return False, "NO_REQUEST", case_prefix

    request_data = json.loads(request_file.read_text())
    instruction = request_data["instruction"]
    completion_mode = request_data["completion_mode"]

    # Load images
    images = {}
    for img_type in ["baseline_main", "baseline_wrist", "current_main", "current_wrist"]:
        img_file = test_dir / f"{case_prefix}-{img_type}.jpg"
        if not img_file.exists():
            return False, f"MISSING_{img_type.upper()}", case_prefix
        images[img_type] = img_file.read_bytes()

    # Call API
    try:
        actual_verdict = await call_verify_api(
            provider, provider_cfg, system_prompt, instruction, completion_mode, images
        )
        passed = actual_verdict == expected_verdict
        return passed, actual_verdict, case_prefix
    except Exception as e:
        return False, f"ERROR: {str(e)[:50]}", case_prefix


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_verify_prompt.py <prompt_file>")
        print("Example: python scripts/test_verify_prompt.py /tmp/verify_improved.md")
        sys.exit(1)

    prompt_file = Path(sys.argv[1])
    if not prompt_file.exists():
        print(f"Prompt file not found: {prompt_file}")
        sys.exit(1)

    system_prompt = prompt_file.read_text()
    provider, provider_cfg = load_config()

    print(f"Testing prompt: {prompt_file}")
    print(f"Provider: {provider}")
    print(f"Model: {provider_cfg['model']}")
    print("=" * 60)

    # Group test files by case
    def group_test_files(test_dir: Path) -> dict[str, Path]:
        """Group test files by case prefix (e.g., rev0001-step0002)."""
        cases = {}
        for request_file in sorted(test_dir.glob("*-request.json")):
            case_prefix = request_file.stem.replace("-request", "")
            cases[case_prefix] = test_dir
        return cases

    continue_dir = TEST_SET_DIR / "continue"
    continue_cases = group_test_files(continue_dir)

    complete_dir = TEST_SET_DIR / "complete"
    complete_cases = group_test_files(complete_dir)

    results = []

    print("\n[CONTINUE cases - should return 'continue']")
    for case_name in sorted(continue_cases.keys()):
        passed, actual, _ = await test_case_by_prefix(
            provider, provider_cfg, system_prompt, continue_dir, case_name, "continue"
        )
        results.append((passed, "continue", actual, case_name))
        status = "✓" if passed else "✗"
        print(f"  {status} {case_name}: expected=continue, actual={actual}")

    print("\n[COMPLETE cases - should return 'completed']")
    for case_name in sorted(complete_cases.keys()):
        passed, actual, _ = await test_case_by_prefix(
            provider, provider_cfg, system_prompt, complete_dir, case_name, "completed"
        )
        results.append((passed, "completed", actual, case_name))
        status = "✓" if passed else "✗"
        print(f"  {case_name}: expected=completed, actual={actual}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r[0])
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} passed, {failed}/{total} failed")

    if failed > 0:
        print("\nFailed cases:")
        for p, expected, actual, name in results:
            if not p:
                print(f"  - {name}: expected={expected}, actual={actual}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
