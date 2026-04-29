#!/usr/bin/env python3
"""
Iterative verify prompt improvement workflow.

Usage:
    python scripts/improve_verify_prompt.py examples/missions/desk_cleanup/verify.md

This script will:
1. Test the current prompt
2. Save each iteration to /tmp/verify_v{N}.md
3. Record success rate for each version
4. Only update the original file when 100% success is achieved
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Import the test logic from test_verify_prompt.py
import importlib.util
spec = importlib.util.spec_from_file_location("test_verify_prompt", "scripts/test_verify_prompt.py")
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_LOG = ROOT / "verify_improvement_log.md"


def log_result(version: int, prompt_path: Path, passed: int, total: int, failed_cases: list):
    """Log test results for a prompt version."""
    success_rate = (passed / total * 100) if total > 0 else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status = "✓ PASS" if passed == total else "✗ FAIL"

    log_entry = f"""
## Version {version} - {status}
- **Timestamp**: {timestamp}
- **File**: `{prompt_path}`
- **Success Rate**: {passed}/{total} ({success_rate:.1f}%)
- **Failed Cases**: {', '.join(failed_cases) if failed_cases else 'None'}

"""

    # Append to log file
    if RESULTS_LOG.exists():
        content = RESULTS_LOG.read_text()
    else:
        content = f"# Verify Prompt Improvement Log\n\nStarted: {timestamp}\n"

    RESULTS_LOG.write_text(content + log_entry)

    print(f"\n{'='*60}")
    print(f"Version {version}: {passed}/{total} passed ({success_rate:.1f}%)")
    if failed_cases:
        print(f"Failed: {', '.join(failed_cases)}")
    print(f"Logged to: {RESULTS_LOG}")
    print(f"{'='*60}\n")


async def test_prompt_version(prompt_path: Path, version: int):
    """Test a prompt version and log results."""
    print(f"\nTesting version {version}: {prompt_path}")

    # Run the test (reuse logic from test_verify_prompt.py)
    # For now, just call the script via subprocess
    import subprocess
    result = subprocess.run(
        ["python", "scripts/test_verify_prompt.py", str(prompt_path)],
        capture_output=True,
        text=True
    )

    # Parse output to get results
    output = result.stdout + result.stderr

    # Extract passed/total from output
    import re
    match = re.search(r"SUMMARY: (\d+)/(\d+) passed", output)
    if match:
        passed = int(match.group(1))
        total = int(match.group(2))
    else:
        print("Could not parse test results")
        return False

    # Extract failed cases
    failed_cases = []
    if "Failed cases:" in output:
        lines = output.split("Failed cases:")[1].split("\n")
        for line in lines:
            if line.strip().startswith("- "):
                case = line.strip()[2:].split(":")[0]
                failed_cases.append(case)

    # Log results
    log_result(version, prompt_path, passed, total, failed_cases)

    return passed == total


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/improve_verify_prompt.py <original_prompt_path>")
        sys.exit(1)

    original_path = Path(sys.argv[1])
    if not original_path.exists():
        print(f"File not found: {original_path}")
        sys.exit(1)

    print(f"Starting iterative improvement for: {original_path}")
    print(f"Results will be logged to: {RESULTS_LOG}")
    print("\nWorkflow:")
    print("1. Edit prompt manually and save to /tmp/verify_v{N}.md")
    print("2. Run: python scripts/test_verify_prompt.py /tmp/verify_v{N}.md")
    print("3. Check verify_improvement_log.md for results")
    print("4. When 100% success, copy to original file")
    print("\nPress Ctrl+C to exit")


if __name__ == "__main__":
    main()
