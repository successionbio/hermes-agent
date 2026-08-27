"""Opt-in, profile-scoped workspaces for durable messaging sessions.

The workspace is a collision/lifecycle boundary, not an OS sandbox.  Names are
opaque hashes of trusted profile and persistent session identity; the manifest
contains hashes only so a directory listing never exposes a raw chat/session
identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_MANIFEST_NAME = ".hermes-session-workspace.json"
_INSTRUCTIONS_LINK = "AGENTS.md"
_SCHEMA_VERSION = 1


class SessionWorkspaceError(RuntimeError):
    """An enabled eligible session could not be bound safely."""


@dataclass(frozen=True)
class SessionWorkspaceBinding:
    cwd: str
    path: Path
    isolated: bool
    created: bool = False
    migrated_legacy: bool = False


def _workspace_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    terminal = config.get("terminal", {}) if isinstance(config, Mapping) else {}
    if not isinstance(terminal, Mapping):
        return {}
    value = terminal.get("session_workspace", {})
    return value if isinstance(value, Mapping) else {}


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def session_workspace_is_eligible(
    config: Mapping[str, Any] | None,
    *,
    platform: str,
    cron_session: bool = False,
) -> bool:
    cfg = _workspace_config(config)
    if cron_session or not _enabled(cfg.get("enabled", False)):
        return False
    raw_platforms = cfg.get("platforms", ["slack"])
    if not isinstance(raw_platforms, (list, tuple, set, frozenset)):
        raise SessionWorkspaceError(
            "terminal.session_workspace.platforms must be a list"
        )
    eligible = {
        str(item).strip().lower() for item in raw_platforms if str(item).strip()
    }
    return str(platform or "").strip().lower() in eligible


def _fingerprint(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="strict"))
        digest.update(b"\0")
    return digest.hexdigest()


def _lstat_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SessionWorkspaceError(
            f"session workspace {label} is unavailable"
        ) from exc
    if stat.S_ISLNK(info.st_mode):
        raise SessionWorkspaceError(f"session workspace {label} must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise SessionWorkspaceError(f"session workspace {label} is not a directory")


def _mkdir_checked(path: Path, *, label: str, mode: int = 0o700) -> bool:
    created = False
    try:
        path.mkdir(mode=mode)
        created = True
    except FileExistsError:
        pass
    except OSError as exc:
        raise SessionWorkspaceError(
            f"could not create session workspace {label}"
        ) from exc
    _lstat_directory(path, label=label)
    return created


def _write_manifest(
    path: Path, *, profile_fingerprint: str, workspace_fingerprint: str
) -> None:
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "profile_fingerprint": profile_fingerprint,
        "workspace_fingerprint": workspace_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = path / _MANIFEST_NAME
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    except OSError as exc:
        raise SessionWorkspaceError(
            "could not write session workspace manifest"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
            handle.write("\n")
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _read_manifest(path: Path) -> Mapping[str, Any]:
    target = path / _MANIFEST_NAME
    try:
        if target.is_symlink():
            raise SessionWorkspaceError(
                "session workspace manifest must not be a symlink"
            )
        value = json.loads(target.read_text(encoding="utf-8"))
    except SessionWorkspaceError:
        raise
    except Exception as exc:
        raise SessionWorkspaceError(
            "session workspace manifest is missing or invalid"
        ) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != _SCHEMA_VERSION:
        raise SessionWorkspaceError(
            "session workspace manifest has an unsupported schema"
        )
    return value


def _validate_managed_workspace(
    path: Path, *, root: Path, profile_fingerprint: str
) -> Mapping[str, Any]:
    _lstat_directory(path, label="binding")
    try:
        resolved = path.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except Exception as exc:
        raise SessionWorkspaceError(
            "stored session workspace escapes the configured root"
        ) from exc
    manifest = _read_manifest(path)
    workspace_fingerprint = str(manifest.get("workspace_fingerprint") or "")
    if (
        manifest.get("profile_fingerprint") != profile_fingerprint
        or not workspace_fingerprint
        or path.name != workspace_fingerprint
        or path.parent.name != profile_fingerprint
    ):
        raise SessionWorkspaceError(
            "stored session workspace does not match its managed identity"
        )
    return manifest


def _enforce_workspace_mode(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        raise SessionWorkspaceError(
            "could not enforce session workspace permissions"
        ) from exc


def _initialize_expected_workspace(
    path: Path,
    *,
    root: Path,
    profile_fingerprint: str,
    workspace_fingerprint: str,
) -> bool:
    """Create or recover the exact deterministic workspace safely."""
    created = _mkdir_checked(path, label="directory")
    manifest_path = path / _MANIFEST_NAME
    if created:
        _write_manifest(
            path,
            profile_fingerprint=profile_fingerprint,
            workspace_fingerprint=workspace_fingerprint,
        )
    elif not manifest_path.exists():
        try:
            if any(path.iterdir()):
                raise SessionWorkspaceError(
                    "session workspace manifest is missing from a non-empty directory"
                )
        except SessionWorkspaceError:
            raise
        except OSError as exc:
            raise SessionWorkspaceError(
                "session workspace directory could not be inspected"
            ) from exc
        _write_manifest(
            path,
            profile_fingerprint=profile_fingerprint,
            workspace_fingerprint=workspace_fingerprint,
        )

    _validate_managed_workspace(
        path, root=root, profile_fingerprint=profile_fingerprint
    )
    _enforce_workspace_mode(path)
    return created


def _install_instructions_link(workspace: Path, raw_path: Any) -> None:
    text = str(raw_path or "").strip()
    if not text:
        return
    source = Path(text)
    if not source.is_absolute():
        raise SessionWorkspaceError(
            "terminal.session_workspace.instructions_path must be absolute"
        )
    try:
        if source.is_symlink() or not source.is_file():
            raise SessionWorkspaceError(
                "session workspace instructions_path must be a regular file"
            )
        source = source.resolve(strict=True)
    except SessionWorkspaceError:
        raise
    except Exception as exc:
        raise SessionWorkspaceError(
            "session workspace instructions_path is unavailable"
        ) from exc

    link = workspace / _INSTRUCTIONS_LINK
    if link.is_symlink():
        try:
            if link.resolve(strict=True) == source:
                return
        except Exception:
            pass
        raise SessionWorkspaceError("session workspace AGENTS.md link is invalid")
    if link.exists():
        raise SessionWorkspaceError(
            "session workspace AGENTS.md collides with an existing path"
        )
    try:
        link.symlink_to(source)
    except OSError as exc:
        raise SessionWorkspaceError(
            "could not create session workspace AGENTS.md link"
        ) from exc


def resolve_session_workspace(
    *,
    config: Mapping[str, Any] | None,
    profile: str,
    session_id: str,
    platform: str,
    static_cwd: str = "",
    stored_cwd: str | None = None,
    cron_session: bool = False,
    allow_inherited_workspace: bool = False,
) -> SessionWorkspaceBinding:
    """Resolve, validate and create the workspace for one durable session.

    Disabled, ineligible and cron callers retain ``static_cwd`` verbatim.
    Eligible callers fail closed on every invalid binding.  A stored static CWD
    is the one intentional migration: a fresh workspace is created without
    copying scratch.  Compression children may reuse a verified managed parent
    workspace when the gateway has independently verified their lineage.
    """
    static = str(static_cwd or "").strip()
    if not session_workspace_is_eligible(
        config, platform=platform, cron_session=cron_session
    ):
        return SessionWorkspaceBinding(
            cwd=static,
            path=Path(static) if static else Path(),
            isolated=False,
        )

    profile_text = str(profile or "").strip()
    session_text = str(session_id or "").strip()
    if not profile_text or not session_text:
        raise SessionWorkspaceError("eligible session workspace identity is incomplete")

    cfg = _workspace_config(config)
    root_text = str(cfg.get("root") or "").strip()
    root = Path(root_text)
    if not root_text or not root.is_absolute():
        raise SessionWorkspaceError("terminal.session_workspace.root must be absolute")

    # Do not permit a configured symlink root.  Its parent may legitimately
    # contain platform-managed symlinks (e.g. macOS /tmp -> /private/tmp), but
    # the configured boundary itself must remain a real directory.
    if not root.parent.is_dir():
        raise SessionWorkspaceError("session workspace root parent is unavailable")
    _mkdir_checked(root, label="root")

    profile_fingerprint = _fingerprint("profile", profile_text)
    workspace_fingerprint = _fingerprint("workspace", profile_text, session_text)
    profile_dir = root / profile_fingerprint
    workspace = profile_dir / workspace_fingerprint
    _mkdir_checked(profile_dir, label="profile directory")

    stored_text = str(stored_cwd or "").strip()
    migrated_legacy = bool(stored_text and static and stored_text == static)
    if stored_text and not migrated_legacy:
        stored_path = Path(stored_text)
        if not stored_path.is_absolute():
            raise SessionWorkspaceError("stored session workspace must be absolute")
        if stored_path == workspace:
            created = _initialize_expected_workspace(
                stored_path,
                root=root,
                profile_fingerprint=profile_fingerprint,
                workspace_fingerprint=workspace_fingerprint,
            )
            _install_instructions_link(stored_path, cfg.get("instructions_path"))
            return SessionWorkspaceBinding(
                cwd=str(stored_path),
                path=stored_path,
                isolated=True,
                created=created,
            )
        if allow_inherited_workspace:
            _validate_managed_workspace(
                stored_path, root=root, profile_fingerprint=profile_fingerprint
            )
            _enforce_workspace_mode(stored_path)
            _install_instructions_link(stored_path, cfg.get("instructions_path"))
            return SessionWorkspaceBinding(
                cwd=str(stored_path), path=stored_path, isolated=True
            )
        raise SessionWorkspaceError(
            "stored session workspace does not match the current persistent session"
        )

    created = _initialize_expected_workspace(
        workspace,
        root=root,
        profile_fingerprint=profile_fingerprint,
        workspace_fingerprint=workspace_fingerprint,
    )
    _install_instructions_link(workspace, cfg.get("instructions_path"))
    return SessionWorkspaceBinding(
        cwd=str(workspace),
        path=workspace,
        isolated=True,
        created=created,
        migrated_legacy=migrated_legacy,
    )
