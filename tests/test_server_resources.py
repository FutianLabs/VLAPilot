"""Tests for MCP resource handlers that do not require hardware imports."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path

from pydantic import AnyUrl, TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_server_module(monkeypatch):
    """Import the server module with lightweight stubs for heavy runtime deps."""
    executor_stub = types.ModuleType("vla_pi05_piper.executor")
    executor_stub.RTCExecutor = object

    runtime_stub = types.ModuleType("vla_pi05_piper.runtime")
    runtime_stub.VLARuntime = object

    mission_stub = types.ModuleType("vlapilot.core.mission")
    mission_stub.load_task_instructions = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "vla_pi05_piper.executor", executor_stub)
    monkeypatch.setitem(sys.modules, "vla_pi05_piper.runtime", runtime_stub)
    monkeypatch.setitem(sys.modules, "vlapilot.core.mission", mission_stub)
    sys.modules.pop("vla_pi05_piper.server", None)

    return importlib.import_module("vla_pi05_piper.server")


def test_list_resource_templates_returns_empty(monkeypatch):
    """Codex probes resource templates, so the server should answer cleanly."""
    server = _load_server_module(monkeypatch)

    assert asyncio.run(server.list_resource_templates()) == []


def test_read_resource_accepts_anyurl(monkeypatch):
    """The MCP SDK passes AnyUrl objects, not plain strings, to read_resource."""
    server = _load_server_module(monkeypatch)
    uri = TypeAdapter(AnyUrl).validate_python("vla://tasks/vocabulary")

    content = asyncio.run(server.read_resource(uri))

    assert "Put the pen into the pen holder." in content


def test_read_resource_redacts_all_api_keys_without_mutating_config(monkeypatch):
    """The config resource should hide planner and verifier API keys."""
    server = _load_server_module(monkeypatch)
    server._config = {
        "agent": {
            "planner": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "planner-secret",
            },
        },
        "arm": {
            "verify": {
                "provider": "local",
                "providers": {
                    "local": {
                        "api_base": "http://127.0.0.1:1821/v1",
                        "api_key": "local-secret",
                    },
                    "openrouter": {
                        "api_base": "https://openrouter.ai/api/v1",
                        "api_key": "router-secret",
                    },
                    "ark": {
                        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
                        "api_key": "ark-secret",
                    },
                },
            },
        }
    }

    content = asyncio.run(server.read_resource("vla://config/current"))

    assert '"api_key": "***"' in content
    assert "planner-secret" not in content
    assert "local-secret" not in content
    assert "router-secret" not in content
    assert "ark-secret" not in content
    assert server._config["agent"]["planner"]["api_key"] == "planner-secret"
    assert server._config["arm"]["verify"]["providers"]["local"]["api_key"] == "local-secret"
    assert server._config["arm"]["verify"]["providers"]["openrouter"]["api_key"] == "router-secret"
    assert server._config["arm"]["verify"]["providers"]["ark"]["api_key"] == "ark-secret"
