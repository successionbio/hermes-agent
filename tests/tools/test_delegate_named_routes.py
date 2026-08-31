"""Trusted named provider/model routes for delegate_task."""

from unittest.mock import patch

import pytest

from tools.delegate_tool import (
    _build_dynamic_schema_overrides,
    _configured_delegation_route_names,
    _resolve_named_delegation_route,
)


BASE_CONFIG = {
    "provider": "openai-codex",
    "model": "gpt-5.6-terra",
    "max_iterations": 50,
    "routes": {
        "luna-cleanup": {
            "description": "Fixed-source cleanup only",
            "model": "gpt-5.6-luna",
            "toolsets": ["file", "code_execution"],
        },
    },
}


def test_configured_route_names_expose_only_valid_mapping_entries():
    cfg = {
        "routes": {
            " z-route ": {"model": "z"},
            "a-route": {"model": "a"},
            "invalid": "not-a-mapping",
            "": {"model": "empty"},
        }
    }

    assert _configured_delegation_route_names(cfg) == ["a-route", "z-route"]


def test_named_route_inherits_provider_and_overrides_model():
    resolved = _resolve_named_delegation_route("luna-cleanup", BASE_CONFIG)

    assert resolved["provider"] == "openai-codex"
    assert resolved["model"] == "gpt-5.6-luna"
    assert resolved["max_iterations"] == 50
    assert resolved["toolsets"] == ["file", "code_execution"]
    assert "routes" not in resolved


def test_unknown_route_fails_with_configured_aliases():
    with pytest.raises(ValueError, match="Unknown delegation route 'cheap'.*luna-cleanup"):
        _resolve_named_delegation_route("cheap", BASE_CONFIG)


def test_dynamic_schema_lists_only_configured_route_aliases():
    with patch("tools.delegate_tool._load_config", return_value=BASE_CONFIG):
        schema = _build_dynamic_schema_overrides()

    route = schema["parameters"]["properties"]["route"]
    assert route["enum"] == ["luna-cleanup"]
    assert "provider/model policy" in route["description"]
    assert "luna-cleanup: Fixed-source cleanup only" in route["description"]


def test_dynamic_schema_omits_route_when_none_are_configured():
    with patch("tools.delegate_tool._load_config", return_value={}):
        schema = _build_dynamic_schema_overrides()

    assert "route" not in schema["parameters"]["properties"]
