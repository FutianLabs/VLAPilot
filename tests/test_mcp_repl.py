"""Tests for the interactive MCP client script."""

from __future__ import annotations

import base64
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import sys

from mcp import types


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mcp_repl.py"
SPEC = importlib.util.spec_from_file_location("mcp_repl", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _call_result(*content: types.ContentBlock) -> types.CallToolResult:
    return types.CallToolResult(content=list(content), isError=False)


def test_parse_cameras_defaults():
    assert MODULE.parse_cameras(None) == ["main", "wrist"]
    assert MODULE.parse_cameras("main") == ["main"]
    assert MODULE.parse_cameras("main,wrist") == ["main", "wrist"]


def test_parse_instruction_mode():
    assert MODULE.parse_instruction_mode(["Close", "the", "laptop", "lid."]) == (
        "Close the laptop lid.",
        "single",
    )
    assert MODULE.parse_instruction_mode(["Close the laptop lid.", "until_clear"]) == (
        "Close the laptop lid.",
        "until_clear",
    )


def test_save_image_blocks(tmp_path: Path):
    payload = base64.b64encode(b"jpeg-bytes").decode()
    result = _call_result(
        types.ImageContent(type="image", data=payload, mimeType="image/jpeg"),
        types.ImageContent(type="image", data=payload, mimeType="image/png"),
    )

    paths = MODULE.save_image_blocks(result, tmp_path, "observe")

    assert [path.suffix for path in paths] == [".jpg", ".png"]
    assert paths[0].read_bytes() == b"jpeg-bytes"
    assert paths[1].read_bytes() == b"jpeg-bytes"


def test_build_server_params_includes_repo_pythonpath():
    args = Namespace(
        python_bin="/tmp/python",
        server="arm",
        server_module=None,
        config=Path("/tmp/config.json"),
        save_dir=Path("/tmp/out"),
        cwd=Path("/tmp"),
    )

    params = MODULE.build_server_params(args)

    assert params.command == "/tmp/python"
    assert params.args == ["-m", "vla_pi05_piper.server", "--config", "/tmp/config.json"]
    assert str(Path(__file__).resolve().parents[1] / "src") in params.env["PYTHONPATH"]


def test_build_server_params_supports_agent_server():
    args = Namespace(
        python_bin="/tmp/python",
        server="agent",
        server_module=None,
        config=Path("/tmp/config.json"),
        save_dir=Path("/tmp/out"),
        cwd=Path("/tmp"),
    )

    params = MODULE.build_server_params(args)

    assert params.args == ["-m", "vlapilot.interface.mcp_server", "--config", "/tmp/config.json"]


def test_validate_config_shape_accepts_new_schema(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"agent": {}, "arm": {}}), encoding="utf-8")

    MODULE.validate_config_shape(config_path)


def test_validate_config_shape_rejects_legacy_schema(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"policy_path": "/tmp/policy", "verify": {}, "arms": {}}),
        encoding="utf-8",
    )

    try:
        MODULE.validate_config_shape(config_path)
    except ValueError as e:
        message = str(e)
    else:
        raise AssertionError("expected ValueError")

    assert "Legacy config schema detected" in message
    assert "policy_path" in message
