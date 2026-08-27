from __future__ import annotations

import asyncio
import json
import os
import stat
import sys

import pytest

from agent.runtime_cwd import (
    SessionCwdUnavailableError,
    get_session_cwd_override,
    resolve_agent_cwd,
)
from gateway.config import Platform
from gateway.session import SessionContext, SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from gateway.session_context import get_session_env
from gateway.session_workspace import (
    SessionWorkspaceError,
    resolve_session_workspace,
)
from tools.delegate_tool import _resolve_workspace_hint


def _config(root, *, enabled=True, platforms=("slack",), instructions_path=""):
    return {
        "terminal": {
            "cwd": "/shared/static",
            "session_workspace": {
                "enabled": enabled,
                "root": str(root),
                "platforms": list(platforms),
                "instructions_path": str(instructions_path),
            },
        }
    }


def _resolve(tmp_path, *, profile="campaign", session_id="session-a", **kwargs):
    root = tmp_path / "workspaces"
    return resolve_session_workspace(
        config=_config(root),
        profile=profile,
        session_id=session_id,
        platform="slack",
        static_cwd="/shared/static",
        **kwargs,
    )


def test_stable_opaque_profile_scoped_mode_0700_workspace(tmp_path):
    first = _resolve(tmp_path)
    again = _resolve(tmp_path)
    other_session = _resolve(tmp_path, session_id="session-b")
    other_profile = _resolve(tmp_path, profile="client-operations")

    assert first.cwd == again.cwd
    assert first.cwd != other_session.cwd
    assert first.cwd != other_profile.cwd
    assert "campaign" not in first.cwd
    assert "session-a" not in first.cwd
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(first.cwd).st_mode) == 0o700

    manifest = json.loads(
        (first.path / ".hermes-session-workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["profile_fingerprint"] not in {"campaign", "session-a"}
    assert "session" not in json.dumps(manifest).lower()


def test_new_persistent_session_identity_gets_fresh_workspace(tmp_path):
    old = _resolve(tmp_path, session_id="old-id")
    new = _resolve(tmp_path, session_id="new-id")
    assert new.cwd != old.cwd
    assert new.migrated_legacy is False


def test_legacy_static_cwd_migrates_without_copying_scratch(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "scratch.txt").write_text("do not copy", encoding="utf-8")
    binding = resolve_session_workspace(
        config=_config(tmp_path / "workspaces"),
        profile="campaign",
        session_id="new-id",
        platform="slack",
        stored_cwd=str(legacy),
        static_cwd=str(legacy),
    )
    assert binding.migrated_legacy is True
    assert not (binding.path / "scratch.txt").exists()


def test_compression_child_may_reuse_verified_parent_workspace(tmp_path):
    parent = _resolve(tmp_path, session_id="parent")
    child = _resolve(
        tmp_path,
        session_id="compressed-child",
        stored_cwd=parent.cwd,
        allow_inherited_workspace=True,
    )
    assert child.cwd == parent.cwd


def _assert_symlink_or_non_directory_collision_fails_closed(tmp_path, collision):
    root = tmp_path / "workspaces"
    config = _config(root)
    if collision == "root":
        target = tmp_path / "outside"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
        with pytest.raises(SessionWorkspaceError):
            resolve_session_workspace(
                config=config,
                profile="campaign",
                session_id="session-a",
                platform="slack",
                static_cwd="/shared/static",
            )
        return

    root.mkdir()
    expected = resolve_session_workspace(
        config=config,
        profile="campaign",
        session_id="session-a",
        platform="slack",
        static_cwd="/shared/static",
    )
    # Rebuild the selected component as an unsafe collision.
    if collision == "profile":
        workspace = expected.path
        (workspace / ".hermes-session-workspace.json").unlink()
        workspace.rmdir()
        profile_dir = workspace.parent
        profile_dir.rmdir()
        target = tmp_path / "outside-profile"
        target.mkdir()
        profile_dir.symlink_to(target, target_is_directory=True)
    else:
        workspace = expected.path
        (workspace / ".hermes-session-workspace.json").unlink()
        workspace.rmdir()
        target = tmp_path / "outside-workspace"
        target.mkdir()
        workspace.symlink_to(target, target_is_directory=True)

    with pytest.raises(SessionWorkspaceError):
        resolve_session_workspace(
            config=config,
            profile="campaign",
            session_id="session-a",
            platform="slack",
            static_cwd="/shared/static",
        )


@pytest.mark.parametrize("collision", ["root", "profile", "workspace"])
@pytest.mark.linux_only
def test_linux_symlink_or_non_directory_collisions_fail_closed(tmp_path, collision):
    _assert_symlink_or_non_directory_collision_fails_closed(tmp_path, collision)


@pytest.mark.parametrize("collision", ["root", "profile", "workspace"])
@pytest.mark.macos_only
def test_macos_symlink_or_non_directory_collisions_fail_closed(tmp_path, collision):
    _assert_symlink_or_non_directory_collision_fails_closed(tmp_path, collision)


def test_disabled_ineligible_and_cron_keep_static_cwd(tmp_path):
    root = tmp_path / "workspaces"
    cases = [
        (_config(root, enabled=False), "slack", False),
        (_config(root), "telegram", False),
        (_config(root), "slack", True),
    ]
    for config, platform, cron_session in cases:
        binding = resolve_session_workspace(
            config=config,
            profile="campaign",
            session_id="session-a",
            platform=platform,
            static_cwd="/shared/static",
            cron_session=cron_session,
        )
        assert binding.cwd == "/shared/static"
        assert binding.isolated is False
    assert not root.exists()


def test_enabled_invalid_platform_config_fails_closed(tmp_path):
    config = _config(tmp_path / "workspaces")
    config["terminal"]["session_workspace"]["platforms"] = "slack"
    with pytest.raises(SessionWorkspaceError, match="must be a list"):
        resolve_session_workspace(
            config=config,
            profile="campaign",
            session_id="session-a",
            platform="slack",
            static_cwd="/shared/static",
        )


def test_invalid_root_and_unexpected_stored_binding_fail_closed(tmp_path):
    with pytest.raises(SessionWorkspaceError, match="absolute"):
        resolve_session_workspace(
            config=_config("relative/root"),
            profile="campaign",
            session_id="session-a",
            platform="slack",
            static_cwd="/shared/static",
        )

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with pytest.raises(SessionWorkspaceError, match="stored"):
        _resolve(tmp_path, stored_cwd=str(foreign))

    root_collision = tmp_path / "root-collision"
    root_collision.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SessionWorkspaceError, match="not a directory"):
        resolve_session_workspace(
            config=_config(root_collision),
            profile="campaign",
            session_id="session-a",
            platform="slack",
            static_cwd="/shared/static",
        )


def test_empty_interrupted_workspace_initialization_recovers_but_nonempty_does_not(
    tmp_path,
):
    binding = _resolve(tmp_path)
    manifest = binding.path / ".hermes-session-workspace.json"
    manifest.unlink()

    recovered = _resolve(tmp_path)
    assert recovered.cwd == binding.cwd
    assert manifest.is_file()

    manifest.unlink()
    (binding.path / "partial-output.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(SessionWorkspaceError, match="non-empty"):
        _resolve(tmp_path)


def test_deleted_expected_stored_workspace_is_rehydrated(tmp_path):
    binding = _resolve(tmp_path)
    (binding.path / ".hermes-session-workspace.json").unlink()
    binding.path.rmdir()

    restored = _resolve(tmp_path, stored_cwd=binding.cwd)
    assert restored.cwd == binding.cwd
    assert restored.created is True
    assert (restored.path / ".hermes-session-workspace.json").is_file()


def test_invalid_inherited_workspace_is_rejected_before_permissions_change(tmp_path):
    foreign = tmp_path / "foreign-inherited"
    foreign.mkdir(mode=0o755)
    before = stat.S_IMODE(foreign.stat().st_mode)

    with pytest.raises(SessionWorkspaceError, match="escapes"):
        _resolve(
            tmp_path,
            session_id="compressed-child",
            stored_cwd=str(foreign),
            allow_inherited_workspace=True,
        )

    assert stat.S_IMODE(foreign.stat().st_mode) == before


def test_optional_instruction_link_is_safe_and_workspace_local(tmp_path):
    instructions = tmp_path / "repo-AGENTS.md"
    instructions.write_text("repository rules", encoding="utf-8")
    binding = resolve_session_workspace(
        config=_config(tmp_path / "workspaces", instructions_path=instructions),
        profile="campaign",
        session_id="session-a",
        platform="slack",
        static_cwd="/shared/static",
    )
    link = binding.path / "AGENTS.md"
    assert link.is_symlink()
    assert link.resolve() == instructions.resolve()


def test_concurrent_contexts_and_delegate_hint_use_task_local_cwd(
    tmp_path, monkeypatch
):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "global"))

    async def read(bound):
        tokens = set_session_vars(session_id=bound.name, cwd=str(bound))
        try:
            await asyncio.sleep(0)
            return get_session_cwd_override(), _resolve_workspace_hint(object())
        finally:
            clear_session_vars(tokens)

    async def main():
        return await asyncio.gather(read(a), read(b))

    results = asyncio.run(main())
    assert results == [(str(a), str(a)), (str(b), str(b))]


def test_session_context_round_trips_bound_cwd():
    context = SessionContext(
        source=SessionSource(platform=Platform.SLACK, chat_id="C1"),
        connected_platforms=[Platform.SLACK],
        home_channels={},
        session_id="sid",
        cwd="/private/workspace",
        cwd_required=True,
    )
    payload = context.to_dict()
    assert payload["cwd"] == "/private/workspace"
    assert payload["cwd_required"] is True


def test_managed_workspace_identity_is_task_local_and_cleared(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tokens = set_session_vars(session_id="sid", cwd=str(workspace), cwd_required=True)
    try:
        assert get_session_env("HERMES_SESSION_WORKSPACE") == str(workspace)
    finally:
        clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_WORKSPACE") == ""


def test_required_workspace_disappearance_never_falls_back(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(fallback))
    tokens = set_session_vars(session_id="sid", cwd=str(workspace), cwd_required=True)
    workspace.rmdir()
    try:
        with pytest.raises(SessionCwdUnavailableError):
            resolve_agent_cwd()
    finally:
        clear_session_vars(tokens)


class _FakeAsyncSessionDB:
    def __init__(self, rows):
        self.rows = rows
        self.updates = []

    async def get_session(self, session_id):
        row = self.rows.get(session_id)
        return dict(row) if row else None

    async def update_session_cwd(self, session_id, cwd, **kwargs):
        if session_id not in self.rows:
            return None
        self.rows[session_id]["cwd"] = cwd
        self.updates.append((session_id, cwd, kwargs))
        return len(self.updates)


def _slack_context(session_id="sid"):
    return SessionContext(
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C1",
            thread_id="T1",
            profile="campaign",
        ),
        connected_platforms=[Platform.SLACK],
        home_channels={},
        session_key="slack:C1:T1",
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_gateway_persists_and_restores_workspace_after_restart(
    tmp_path, monkeypatch
):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    config = _config(tmp_path / "workspaces")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    rows = {"sid": {"id": "sid", "cwd": None, "parent_session_id": None}}
    db = _FakeAsyncSessionDB(rows)

    first_runner = object.__new__(GatewayRunner)
    first_runner._session_db = db
    first_runner._active_profile_name = lambda: "campaign"
    first_context = _slack_context()
    first = await first_runner._bind_session_workspace(first_context)

    restarted_runner = object.__new__(GatewayRunner)
    restarted_runner._session_db = db
    restarted_runner._active_profile_name = lambda: "campaign"
    restarted_context = _slack_context()
    restored = await restarted_runner._bind_session_workspace(restarted_context)

    assert restored == first
    assert restarted_context.cwd_required is True
    assert db.rows["sid"]["cwd"] == first
    assert len(db.updates) == 1


@pytest.mark.asyncio
async def test_gateway_compression_child_keeps_parent_workspace(tmp_path, monkeypatch):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    config = _config(tmp_path / "workspaces")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    parent = resolve_session_workspace(
        config=config,
        profile="campaign",
        session_id="parent",
        platform="slack",
        static_cwd="/shared/static",
    )
    rows = {
        "parent": {
            "id": "parent",
            "cwd": parent.cwd,
            "ended_at": 1,
            "end_reason": "compression",
            "parent_session_id": None,
        },
        "child": {
            "id": "child",
            "cwd": parent.cwd,
            "parent_session_id": "parent",
        },
    }
    db = _FakeAsyncSessionDB(rows)
    runner = object.__new__(GatewayRunner)
    runner._session_db = db
    runner._active_profile_name = lambda: "campaign"
    context = _slack_context("child")

    assert await runner._bind_session_workspace(context) == parent.cwd
    assert db.updates == []


@pytest.mark.asyncio
async def test_gateway_compression_child_rejects_a_cwd_not_bound_to_its_parent(
    tmp_path, monkeypatch
):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    config = _config(tmp_path / "workspaces")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    parent = resolve_session_workspace(
        config=config,
        profile="campaign",
        session_id="parent",
        platform="slack",
        static_cwd="/shared/static",
    )
    unrelated = resolve_session_workspace(
        config=config,
        profile="campaign",
        session_id="unrelated",
        platform="slack",
        static_cwd="/shared/static",
    )
    rows = {
        "parent": {
            "id": "parent",
            "cwd": parent.cwd,
            "ended_at": 1,
            "end_reason": "compression",
            "parent_session_id": None,
        },
        "child": {
            "id": "child",
            "cwd": unrelated.cwd,
            "parent_session_id": "parent",
        },
    }
    runner = object.__new__(GatewayRunner)
    runner._session_db = _FakeAsyncSessionDB(rows)
    runner._active_profile_name = lambda: "campaign"

    with pytest.raises(SessionWorkspaceError, match="does not match"):
        await runner._bind_session_workspace(_slack_context("child"))


@pytest.mark.asyncio
async def test_gateway_binding_seeds_terminal_file_and_code_cwds_independently(
    tmp_path,
):
    from gateway.run import GatewayRunner
    from tools import code_execution_tool, file_tools, terminal_tool

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    contexts = []
    for name in ("one", "two"):
        workspace = tmp_path / name
        workspace.mkdir()
        context = _slack_context(name)
        context.cwd = str(workspace)
        context.cwd_required = True
        contexts.append((context, workspace))

    async def resolve(context, workspace):
        tokens = runner._set_session_env(context)
        try:
            await asyncio.sleep(0)
            return (
                file_tools._resolve_path_for_task(
                    "relative.txt", task_id=context.session_id
                ),
                terminal_tool._resolve_command_cwd(
                    workdir=None,
                    default_cwd="/shared/static",
                    session_key=context.session_id,
                ),
                code_execution_tool._resolve_child_cwd(
                    "project", str(tmp_path / "staging"), context.session_id
                ),
                get_session_env("HERMES_SESSION_WORKSPACE"),
            )
        finally:
            runner._clear_session_env(tokens)

    results = await asyncio.gather(
        *(resolve(context, workspace) for context, workspace in contexts)
    )
    for (context, workspace), result in zip(contexts, results):
        file_path, terminal_cwd, code_cwd, trusted_workspace = result
        assert file_path == workspace / "relative.txt"
        assert terminal_cwd == str(workspace)
        assert code_cwd == str(workspace)
        assert trusted_workspace == str(workspace)
