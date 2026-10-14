"""Isolated managed profiles and saved SSH remotes.

A *profile* is a self-contained executor root: configuration, credentials,
provenance, software latch and the published runtime record.  Profiles are
isolated per run mode and target so that

* a simulation profile can never serve as the real robot, and
* a real profile can never silently fall back to a simulator.

Legacy flat per-user roots (``config.json`` and ``estop.latched`` directly in
the application data directory) are migrated into ``profiles/<mode>/<target>``
while preserving the old restriction (read-only) and the software latch.  A
migration never *relaxes* a restriction: a legacy read-only configuration stays
read-only, a legacy estop latch stays latched, and every known execution or
feedback limit from the legacy configuration is carried over explicitly instead
of falling back to a default.  Identity, storage and listener fields
(``backend``, ``data_dir``, ``host``, ``port``, ``managed_*``) are always
rebuilt from the isolated profile, physical transport/SDK paths are copied only
from a physical legacy source, and credential fields are never carried across
profiles.  Existing files are copied into the profile's ``backup/`` directory
before anything is rewritten, so the originals stay auditable.

Nothing in this module talks to hardware or to the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from .interaction_types import InteractionPolicy
from .models import DomainError, Settings

MODES = ("simulation", "real")
LOCAL_TARGET = "local"
SIMULATION_BACKENDS = ("mujoco", "sim")
REAL_BACKEND = "agx"
REAL_CAN_INTERFACES = ("agx_cando", "socketcan")
CONTROL_PROFILES = ("direct", "calibration")
PROFILE_SCHEMA = 1
MAX_REMOTES = 64

CONFIG_NAME = "config.json"
PROVENANCE_NAME = "provenance.json"
RUNTIME_NAME = "runtime.json"
REMOTES_NAME = "remotes.json"
LATCH_NAME = "estop.latched"
MODEL_TOKEN_NAME = "model.token"
OPERATOR_TOKEN_NAME = "operator.token"
EXECUTOR_LOG_NAME = "executor.log"
INTERACTION_POLICY_NAME = "interaction-policy.json"
BACKUP_DIR_NAME = "backup"
BACKUP_SUFFIX = ".orig"

# Fields that belong to the isolated profile itself.  A legacy listener address,
# data directory, backend or managed identity must never redirect a new profile,
# so these are rebuilt from the mode/target instead of copied.
_NON_MIGRATABLE_FIELDS = frozenset({
    "backend", "data_dir", "host", "port", "managed_profile_id", "managed_control",
})
# Physical transport and physical SDK lookup paths.  They are copied only from a
# legacy configuration whose backend is the physical backend, so a simulator
# config can never point a real profile at a CAN device or a foreign SDK tree.
_PHYSICAL_FIELDS = frozenset({
    "can_interface", "can_channel", "cando_device_index", "firmware_profile",
    "sdk_root", "cando_source",
})
# Simulation-only settings: a real profile never inherits a simulator seed/asset.
_SIMULATION_ONLY_FIELDS = frozenset({"simulation_seed", "simulation_asset"})
_INTERACTION_POLICY_FIELD = "interaction_policy"
# Managed identity fields are enforced by _repair_config, never copied from a
# legacy file and never preserved from an existing profile.
_MANAGED_FIELDS = frozenset({"managed_control", "managed_profile_id"})
_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Letters, digits, dot, underscore, dash, colon, at and IPv6 brackets only.
_HOST_RE = re.compile(r"^[A-Za-z0-9\[][A-Za-z0-9._:@\[\]-]{0,254}$")
_RESERVED_REMOTE_NAMES = frozenset({LOCAL_TARGET})


class ProfileError(DomainError, ValueError):
    """Structured profile failure that is also a ``ValueError`` for library callers."""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(code, message, status)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_mode(mode: Any) -> str:
    if mode not in MODES:
        raise ProfileError("invalid_mode", "Run mode must be 'simulation' or 'real'.")
    return mode


def _validate_component(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise ProfileError(f"invalid_{what}", f"{what.capitalize()} must be a string.")
    text = value.strip()
    if not _NAME_RE.fullmatch(text):
        raise ProfileError(
            f"invalid_{what}",
            f"{what.capitalize()} must start with a letter or digit and contain only letters, digits, '.', '_' or '-' (max 64 characters).")
    return text


def validate_target(target: Any) -> str:
    """A target is ``local`` or a saved remote name; used as a directory component."""
    return _validate_component(target, "target")


def validate_remote_name(name: Any) -> str:
    text = _validate_component(name, "remote_name")
    if text.lower() in _RESERVED_REMOTE_NAMES:
        raise ProfileError("invalid_remote_name", f"{text!r} is reserved for the local target.")
    return text


def validate_ssh_host(ssh_host: Any) -> str:
    """Accept an SSH config alias or hostname, never an option or a credential.

    The value is always passed as a single ``ssh`` argument, but leading dashes
    are rejected anyway to rule out option injection, and spaces/control
    characters are rejected so nothing can be smuggled into a remote command.
    Passwords and passphrases are never stored; ``user:secret@host`` is refused.
    """
    if not isinstance(ssh_host, str):
        raise ProfileError("invalid_ssh_host", "SSH host must be a string.")
    text = ssh_host.strip()
    if not text:
        raise ProfileError("invalid_ssh_host", "SSH host must not be empty.")
    if len(text) > 255:
        raise ProfileError("invalid_ssh_host", "SSH host is too long.")
    if any(ord(char) < 33 or ord(char) == 127 for char in text):
        raise ProfileError("invalid_ssh_host", "SSH host must not contain spaces or control characters.")
    if text.startswith("-"):
        raise ProfileError("invalid_ssh_host", "SSH host must not start with '-' (option injection).")
    if not _HOST_RE.fullmatch(text):
        raise ProfileError(
            "invalid_ssh_host",
            "SSH host may contain only letters, digits, '.', '_', '-', ':', '@' and brackets.")
    if text.count("@") > 1:
        raise ProfileError("invalid_ssh_host", "SSH host must contain at most one '@'.")
    if "@" in text:
        user, destination = text.rsplit("@", 1)
        if not user or not destination:
            raise ProfileError("invalid_ssh_host", "SSH host must name a user and a destination.")
        if ":" in user:
            raise ProfileError("password_not_allowed", "Never store passwords or passphrases; use SSH keys and an agent.")
    return text


def profile_id(mode: str, target: str = LOCAL_TARGET) -> str:
    """Stable profile identity for a mode/target pair.

    It deliberately does not depend on the base directory so a profile keeps
    its identity if the per-user data directory is moved, while a remote
    executor still reports a distinct identity for its own local profile.
    """
    material = f"piperx-profile\x00{validate_mode(mode)}\x00{validate_target(target)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def profile_root(base, mode: str, target: str = LOCAL_TARGET) -> Path:
    """``<base>/profiles/<mode>/<target>`` with an escape check."""
    base_path = Path(base).expanduser()
    mode = validate_mode(mode)
    target = validate_target(target)
    root = base_path / "profiles" / mode / target
    expected = (base_path.resolve() / "profiles" / mode / target)
    if root.resolve() != expected:
        raise ProfileError("invalid_target", "Profile path escapes the application data directory.")
    return root


@dataclass(frozen=True)
class Profile:
    root: Path
    mode: str
    target: str
    profile_id: str
    backend: str
    readonly: bool
    latch: bool
    config_path: Path
    provenance_path: Path
    runtime_path: Path
    model_token_file: Path
    operator_token_file: Path
    log_path: Path

    @property
    def startup_lock_path(self) -> Path:
        return self.root / "startup.lock"

    @property
    def data_dir(self) -> Path:
        return self.root

    @property
    def managed_control(self) -> bool:
        return True


# --------------------------------------------------------------------------
# small filesystem helpers
# --------------------------------------------------------------------------

def _tighten(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Windows only implements the read-only bit; the parent 0700 directory
        # is the real protection there.
        pass


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _tighten(path, 0o700)


def _atomic_write(path: Path, content: str | bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        if isinstance(content, bytes):
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        _tighten(Path(temporary), mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _atomic_write_json(path: Path, data: dict, mode: int = 0o600) -> None:
    _atomic_write(path, json.dumps(data, indent=2, allow_nan=False) + "\n", mode)


def _read_json(path: Path, code: str = "invalid_profile_config") -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(code, f"{path.name} is not readable JSON ({type(exc).__name__}); refusing to guess.") from None


def _read_optional_json(path: Path | None, code: str = "invalid_profile_config") -> Any:
    if path is None or not path.is_file():
        return None
    return _read_json(path, code)


def _read_optional_object(path: Path | None, code: str) -> dict | None:
    """Read an optional JSON configuration object; anything else is refused."""
    if path is None or not path.is_file():
        return None
    data = _read_json(path, code)
    if not isinstance(data, dict):
        raise ProfileError(code, f"{path.name} must contain a JSON object; refusing to migrate from it.")
    return data


def _backup_existing(source: Path | None, root: Path, *, label: str | None = None) -> Path | None:
    """Copy an existing file byte-for-byte into ``<root>/backup`` before a rewrite.

    The original is never touched.  The backup keeps the exact bytes (so an
    operator can compare it with the source later) and is written 0600 inside the
    0700 profile root.  Returns ``None`` when there is nothing to back up.
    """
    if source is None or not source.is_file():
        return None
    target = root / BACKUP_DIR_NAME / f"{label or source.name}{BACKUP_SUFFIX}"
    try:
        payload = source.read_bytes()
    except OSError:
        raise ProfileError("backup_failed",
                           f"{source.name} could not be read for backup; refusing to modify it.") from None
    _atomic_write(target, payload, 0o600)
    return target


def _publish_new_token(path: Path) -> None:
    """Atomically publish a new credential; a concurrent creator wins cleanly."""
    content = secrets.token_urlsafe(48) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _tighten(Path(temporary), 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass  # another writer published first; its credential wins
        except (OSError, NotImplementedError):
            # Filesystems without hard links: exclusive create, never overwrite.
            try:
                handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(handle, "w", encoding="utf-8") as sink:
                    sink.write(content)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _ensure_token(path: Path) -> Path:
    """Create a fresh 0600 credential exactly once; never rotate implicitly."""
    deadline = time.monotonic() + 2.0
    while True:
        if path.is_file():
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError:
                raise ProfileError("invalid_credentials", f"Credential file {path.name} is unreadable.") from None
            if len(token) >= 32 and not any(char.isspace() for char in token):
                _tighten(path, 0o600)
                return path
            if time.monotonic() >= deadline:
                raise ProfileError("invalid_credentials",
                                   f"Credential file {path.name} is invalid; refusing to rotate a live credential.")
            time.sleep(0.005)  # a concurrent starter may still be publishing
            continue
        _publish_new_token(path)
        if time.monotonic() >= deadline:
            raise ProfileError("invalid_credentials", f"Credential file {path.name} could not be created.")


def _legacy_paths(base: Path) -> tuple[Path, Path]:
    return base / CONFIG_NAME, base / LATCH_NAME


def _legacy_policy_path(base: Path) -> Path:
    return base / INTERACTION_POLICY_NAME


# --------------------------------------------------------------------------
# configuration construction / validation
# --------------------------------------------------------------------------

def _annotation_mentions_path(annotation: Any) -> bool:
    if annotation is Path:
        return True
    return any(_annotation_mentions_path(argument) for argument in get_args(annotation))


# Path-typed Settings fields (``simulation_asset``, ``sdk_root``, ...).  A JSON
# string is the only acceptable external form; it stays a string in the config.
_PATH_FIELDS = frozenset(
    name for name, field in Settings.model_fields.items() if _annotation_mentions_path(field.annotation))


def _migratable_field_names(mode: str, *, physical_source: bool) -> frozenset[str]:
    """Known Settings fields a legacy config may contribute to this profile.

    Everything is preserved by default so a newly added limit is never silently
    dropped; only profile identity/storage/listener fields and mode-inapplicable
    physical settings are excluded.
    """
    names = set(Settings.model_fields) - _NON_MIGRATABLE_FIELDS
    names.discard(_INTERACTION_POLICY_FIELD)
    if mode != "simulation":
        names -= _SIMULATION_ONLY_FIELDS
    if not (mode == "real" and physical_source):
        names -= _PHYSICAL_FIELDS
    return frozenset(names)


def _first_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid value"
    error = errors[0]
    location = ".".join(str(part) for part in error.get("loc", ())) or "value"
    return f"{location}: {error.get('msg', 'invalid value')}"


def _validate_present_value(name: str, value: Any, *, code: str) -> None:
    """Validate one configured value against its Settings field, strictly.

    A malformed value is refused.  It is never coerced, dropped or replaced by a
    default: normalizing a restriction towards a permissive default is exactly
    the failure this module must prevent.
    """
    field = Settings.model_fields.get(name)
    if field is None:
        raise ProfileError(code, f"Unrecognized configuration field {name!r}; refusing to guess its meaning.")
    candidate = Path(value) if name in _PATH_FIELDS and isinstance(value, str) else value
    try:
        Settings.model_validate({name: candidate}, strict=True)
    except ValidationError as exc:
        raise ProfileError(
            code,
            f"Configuration field {name!r} is invalid ({_first_validation_error(exc)}); refusing to normalize it.") from None


def _validated_policy(policy: Any, *, code: str) -> dict:
    """Validate an interaction policy without loosening any of its limits."""
    if not isinstance(policy, dict):
        raise ProfileError(code, f"{_INTERACTION_POLICY_FIELD} must be a JSON object.")
    try:
        validated = InteractionPolicy.model_validate(policy, strict=True)
    except ValidationError as exc:
        raise ProfileError(
            code,
            f"Interaction policy is invalid ({_first_validation_error(exc)}); refusing to loosen it.") from None
    return validated.model_dump(mode="json")


def _default_can_settings() -> dict:
    if sys.platform.startswith("linux"):
        return {"can_interface": "socketcan", "can_channel": "can0"}
    return {"can_interface": "agx_cando", "can_channel": "0"}


def _default_config(mode: str, backend: str, root: Path, identity: str) -> dict:
    """A genuinely new profile: conservative Settings defaults, written explicitly.

    Every applicable limit is spelled out so a later migration or repair can
    never mistake an unset field for permission to pick a looser value.
    """
    defaults = Settings().model_dump(mode="json")
    defaults.update(_default_can_settings())  # platform-appropriate transport defaults
    migratable = _migratable_field_names(mode, physical_source=(mode == "real"))
    config: dict[str, Any] = {
        name: defaults[name]
        for name in Settings.model_fields  # stable, auditable field order
        if name in migratable and defaults.get(name) is not None
    }
    config.update({
        "backend": backend,
        "host": "127.0.0.1",
        "port": 0,  # internal automatic binding; the real URL is published in runtime.json
        "data_dir": str(root),
        "managed_control": True,
        "managed_profile_id": identity,
    })
    return config


def _config_from_legacy(mode: str, backend: str, legacy: Any, root: Path, identity: str) -> tuple[dict, bool]:
    """Build a new managed profile config, preserving every legacy restriction.

    All known Settings fields that describe an execution/feedback limit, the
    control profile, mode-specific simulation settings, physical transport/SDK
    paths and an embedded interaction policy are copied verbatim (after strict
    validation).  Identity, storage and listener fields are rebuilt, physical
    paths are copied only from a physical legacy source, and credential-looking
    keys are not Settings fields so they can never be copied.
    """
    config = _default_config(mode, backend, root, identity)
    if legacy is None:
        return config, False
    if not isinstance(legacy, dict):
        raise ProfileError("invalid_legacy_config", "Legacy config.json must contain a JSON object.")
    if "allow_motion" not in legacy:
        raise ProfileError(
            "invalid_legacy_config",
            "Legacy config.json does not declare allow_motion; an existing configuration is never "
            "assumed permissive.")
    if not isinstance(legacy["allow_motion"], bool):
        raise ProfileError(
            "invalid_legacy_config",
            f"Legacy allow_motion must be true or false, not {legacy['allow_motion']!r}; "
            "refusing to produce a permissive profile.")
    physical_source = legacy.get("backend") == REAL_BACKEND
    for name in sorted(_migratable_field_names(mode, physical_source=physical_source)):
        if name not in legacy:
            continue
        _validate_present_value(name, legacy[name], code="invalid_legacy_config")
        config[name] = legacy[name]  # copied verbatim; never normalized or dropped
    if _INTERACTION_POLICY_FIELD in legacy and legacy[_INTERACTION_POLICY_FIELD] is not None:
        config[_INTERACTION_POLICY_FIELD] = _validated_policy(
            legacy[_INTERACTION_POLICY_FIELD], code="invalid_legacy_config")
    return config, True


def _validate_restriction(config: dict, mode: str) -> str:
    """Return the profile backend, refusing any mode-restriction violation."""
    if not isinstance(config, dict):
        raise ProfileError("invalid_profile_config", "Managed profile configuration must be a JSON object.")
    configured = config.get("backend")
    if mode == "real":
        if configured != REAL_BACKEND:
            raise ProfileError(
                "mode_restriction",
                f"Real mode requires backend {REAL_BACKEND!r}; this profile declares {configured!r}. "
                "Refusing to substitute a simulator for the real robot.")
        return REAL_BACKEND
    if configured not in SIMULATION_BACKENDS:
        raise ProfileError(
            "mode_restriction",
            f"Simulation mode cannot use backend {configured!r}; use a separate real profile for physical hardware.")
    return configured


def _repair_config(config: dict, mode: str, backend: str, root: Path, identity: str) -> bool:
    """Enforce managed invariants in place; return True when something changed.

    Managed identity/storage fields are rebuilt, but a malformed restriction is
    never rewritten: a hand-edited or corrupt value is refused explicitly rather
    than repaired towards a permissive default.  A missing ``allow_motion`` in an
    existing profile is refused instead of being assumed permissive.
    """
    changed = False
    if config.get("managed_control") is not True:
        config["managed_control"] = True
        changed = True
    if config.get("managed_profile_id") != identity:
        config["managed_profile_id"] = identity
        changed = True
    if str(config.get("data_dir")) != str(root):
        config["data_dir"] = str(root)
        changed = True

    if "allow_motion" not in config:
        raise ProfileError(
            "invalid_profile_config",
            "Managed profile configuration does not declare allow_motion; refusing to assume motion is enabled.")
    if not isinstance(config["allow_motion"], bool):
        raise ProfileError(
            "invalid_profile_config",
            f"Managed profile allow_motion must be true or false, not {config['allow_motion']!r}; "
            "refusing to change it.")

    for name, value in list(config.items()):
        if name in _NON_MIGRATABLE_FIELDS or name in _MANAGED_FIELDS or name in ("host", "port"):
            continue
        if name == _INTERACTION_POLICY_FIELD:
            if value is not None:
                _validated_policy(value, code="invalid_profile_config")
            continue
        _validate_present_value(name, value, code="invalid_profile_config")

    host = config.get("host", "127.0.0.1")
    if host not in _LOOPBACK_HOSTS:
        raise ProfileError("invalid_profile_config",
                           "Managed profiles must bind a loopback listener; use an SSH tunnel for remote access.")
    port = config.get("port", 0)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ProfileError("invalid_profile_config", "Managed profile port must be an integer between 0 and 65535.")
    return changed


def _migrate_latch(root: Path, legacy_latch: Path | None) -> bool:
    """Carry a legacy software latch into the profile; never drop one."""
    if legacy_latch is None or not legacy_latch.exists():
        return False
    target = root / LATCH_NAME
    if target.exists():
        return True
    if not legacy_latch.is_file():
        raise ProfileError("latch_unreadable",
                           f"Legacy software latch {legacy_latch.name} is not a readable file; refusing to drop it silently.")
    try:
        content = legacy_latch.read_text(encoding="utf-8")
    except OSError:
        raise ProfileError("latch_unreadable",
                           f"Legacy software latch {legacy_latch.name} is unreadable; refusing to drop it silently.") from None
    _atomic_write(target, content, 0o600)
    return True


def _check_provenance(provenance: Any, mode: str, target: str, identity: str) -> None:
    if not provenance:
        return
    if not isinstance(provenance, dict):
        raise ProfileError("profile_identity_mismatch", "Profile provenance must be a JSON object.")
    if (provenance.get("profile_id") not in (None, identity)
            or provenance.get("mode") not in (None, mode)
            or provenance.get("target") not in (None, target)):
        raise ProfileError(
            "profile_identity_mismatch",
            f"Profile provenance does not match {mode}/{target}; refusing to reuse a foreign profile directory.")


def _updated_provenance(existing: Any, mode: str, target: str, identity: str, backend: str,
                        config: dict, latch: bool, migrated_from: Path | None,
                        *, legacy_policy: Path | None = None) -> dict:
    provenance = dict(existing) if isinstance(existing, dict) else {}
    provenance.update({
        "schema": PROFILE_SCHEMA,
        "profile_id": identity,
        "mode": mode,
        "target": target,
        "backend": backend,
        "readonly": config.get("allow_motion") is False,
        "latch": bool(latch),
        "managed_control": True,
        "created_at": provenance.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
    })
    if migrated_from is not None:
        provenance["migrated"] = True
        provenance["legacy_source"] = str(migrated_from)
        provenance["migration_readonly"] = config.get("allow_motion") is False
    if legacy_policy is not None:
        provenance["legacy_policy_source"] = str(legacy_policy)
    return provenance


def _build_profile(base: Path, mode: str, target: str, identity: str, backend: str, config: dict,
                   root: Path) -> Profile:
    return Profile(
        root=root,
        mode=mode,
        target=target,
        profile_id=identity,
        backend=backend,
        readonly=config.get("allow_motion") is False,
        latch=(root / LATCH_NAME).is_file(),
        config_path=root / CONFIG_NAME,
        provenance_path=root / PROVENANCE_NAME,
        runtime_path=root / RUNTIME_NAME,
        model_token_file=root / MODEL_TOKEN_NAME,
        operator_token_file=root / OPERATOR_TOKEN_NAME,
        log_path=root / EXECUTOR_LOG_NAME,
    )


def ensure_profile(base, mode: str, target: str = LOCAL_TARGET, *, simulation_backend: str = "mujoco") -> Profile:
    """Provision (or validate) the isolated profile for a mode/target pair."""
    mode = validate_mode(mode)
    target = validate_target(target)
    if mode == "simulation" and simulation_backend not in SIMULATION_BACKENDS:
        raise ProfileError("invalid_backend", "Simulation backend must be 'mujoco' or 'sim'.")
    base_path = Path(base).expanduser()
    root = profile_root(base_path, mode, target)
    _mkdir_private(root)
    identity = profile_id(mode, target)
    config_path = root / CONFIG_NAME
    provenance_path = root / PROVENANCE_NAME

    provenance = _read_optional_json(provenance_path, "profile_identity_mismatch")
    _check_provenance(provenance, mode, target, identity)

    legacy_config: Path | None = None
    legacy_latch: Path | None = None
    legacy_policy: Path | None = None
    migrated_from: Path | None = None
    policy: Any = None
    if target == LOCAL_TARGET:
        legacy_config, legacy_latch = _legacy_paths(base_path)
        legacy_policy = _legacy_policy_path(base_path)

    if config_path.is_file():
        config = _read_json(config_path)
        backend = _validate_restriction(config, mode)
        if _repair_config(config, mode, backend, root, identity):
            _backup_existing(config_path, root)  # snapshot the file before rewriting it
            _atomic_write_json(config_path, config, 0o600)
    else:
        legacy = _read_optional_object(legacy_config, "invalid_legacy_config")
        if legacy is not None:
            migrated_from = legacy_config
        policy = _read_optional_object(legacy_policy, "invalid_legacy_config")
        if policy is not None:
            migrated_from = migrated_from or legacy_policy
        backend = simulation_backend if mode == "simulation" else REAL_BACKEND
        config, _migrated = _config_from_legacy(mode, backend, legacy, root, identity)
        if policy is not None:
            # The standalone operator policy is authoritative; an embedded legacy
            # copy must agree, otherwise choosing either could loosen a limit.
            file_policy = _validated_policy(policy, code="invalid_legacy_config")
            embedded = config.get(_INTERACTION_POLICY_FIELD)
            if embedded is not None and embedded != file_policy:
                raise ProfileError(
                    "invalid_legacy_config",
                    f"Legacy {CONFIG_NAME} and {INTERACTION_POLICY_NAME} declare different interaction policies; "
                    "refusing to choose one and risk loosening a restriction.")
            config[_INTERACTION_POLICY_FIELD] = file_policy
        # Back up every legacy original before publishing anything new.
        _backup_existing(legacy_config, root, label=f"legacy-{CONFIG_NAME}")
        _backup_existing(legacy_policy, root, label=f"legacy-{INTERACTION_POLICY_NAME}")
        _backup_existing(legacy_latch, root, label=f"legacy-{LATCH_NAME}")
        _atomic_write_json(config_path, config, 0o600)
        if policy is not None:
            _atomic_write_json(root / INTERACTION_POLICY_NAME, config[_INTERACTION_POLICY_FIELD], 0o600)
        _migrate_latch(root, legacy_latch)

    _ensure_token(root / MODEL_TOKEN_NAME)
    _ensure_token(root / OPERATOR_TOKEN_NAME)

    profile = _build_profile(base_path, mode, target, identity, backend, config, root)
    _atomic_write_json(
        provenance_path,
        _updated_provenance(provenance, mode, target, identity, backend, config, profile.latch, migrated_from,
                            legacy_policy=legacy_policy if policy is not None else None),
        0o600)
    return profile


def load_profile(base, mode: str, target: str = LOCAL_TARGET) -> Profile:
    """Read an existing profile without creating or repairing anything.

    Restrictions are still validated: a malformed or missing ``allow_motion`` is
    refused here rather than handed to a caller that might treat it as permissive.
    """
    mode = validate_mode(mode)
    target = validate_target(target)
    root = profile_root(base, mode, target)
    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        raise ProfileError("profile_missing", f"Managed profile {mode}/{target} is not initialized.")
    config = _read_json(config_path)
    backend = _validate_restriction(config, mode)
    _repair_config(dict(config), mode, backend, root, profile_id(mode, target))  # validate only; no rewrite
    return _build_profile(Path(base).expanduser(), mode, target, profile_id(mode, target), backend, config, root)


# --------------------------------------------------------------------------
# saved SSH remotes
# --------------------------------------------------------------------------

def _remotes_path(base) -> Path:
    return Path(base).expanduser() / REMOTES_NAME


def _load_remote_store(base) -> dict:
    path = _remotes_path(base)
    if not path.is_file():
        return {"version": 1, "remotes": []}
    data = _read_json(path, "invalid_remote_store")
    if not isinstance(data, dict) or not isinstance(data.get("remotes"), list):
        raise ProfileError("invalid_remote_store", f"{REMOTES_NAME} must contain a 'remotes' list.")
    remotes: list[dict] = []
    for entry in data["remotes"]:
        if not isinstance(entry, dict):
            raise ProfileError("invalid_remote_store", f"{REMOTES_NAME} contains a non-object entry.")
        # Re-validate stored values: a hand-edited file must not reintroduce
        # option injection, credentials, or reserved names.
        remotes.append({
            "name": validate_remote_name(entry.get("name")),
            "ssh_host": validate_ssh_host(entry.get("ssh_host")),
        })
    if len(remotes) > MAX_REMOTES:
        raise ProfileError("invalid_remote_store", f"{REMOTES_NAME} contains too many remotes.")
    return {"version": 1, "remotes": remotes}


def list_remotes(base) -> list[dict]:
    """Return saved remotes as ``[{'name': ..., 'ssh_host': ...}]``."""
    return [dict(remote) for remote in _load_remote_store(base)["remotes"]]


def get_remote(base, name: str) -> dict | None:
    try:
        wanted = validate_remote_name(name)
    except ProfileError:
        return None
    for remote in _load_remote_store(base)["remotes"]:
        if remote["name"] == wanted:
            return dict(remote)
    return None


def save_remote(base, name: str, ssh_host: str) -> dict:
    """Add or update a saved SSH remote; never stores a password."""
    name = validate_remote_name(name)
    ssh_host = validate_ssh_host(ssh_host)
    store = _load_remote_store(base)
    remotes = store["remotes"]
    index = next((i for i, remote in enumerate(remotes) if remote["name"] == name), None)
    if index is None:
        conflict = next((remote for remote in remotes if remote["name"].lower() == name.lower()), None)
        if conflict is not None:
            raise ProfileError("remote_conflict",
                               f"A remote named {conflict['name']!r} already exists; remote names are case-insensitive.")
        if len(remotes) >= MAX_REMOTES:
            raise ProfileError("too_many_remotes", f"At most {MAX_REMOTES} remotes can be saved.")
        remotes.append({"name": name, "ssh_host": ssh_host})
    else:
        remotes[index] = {"name": name, "ssh_host": ssh_host}
    _atomic_write_json(_remotes_path(base), {"version": 1, "remotes": remotes}, 0o600)
    return {"name": name, "ssh_host": ssh_host}


def remove_remote(base, name: str) -> bool:
    name = validate_remote_name(name)
    store = _load_remote_store(base)
    remotes = store["remotes"]
    remaining = [remote for remote in remotes if remote["name"] != name]
    if len(remaining) == len(remotes):
        return False
    _atomic_write_json(_remotes_path(base), {"version": 1, "remotes": remaining}, 0o600)
    return True


__all__ = [
    "MODES", "LOCAL_TARGET", "SIMULATION_BACKENDS", "REAL_BACKEND",
    "CONFIG_NAME", "PROVENANCE_NAME", "RUNTIME_NAME", "REMOTES_NAME", "LATCH_NAME",
    "INTERACTION_POLICY_NAME", "BACKUP_DIR_NAME",
    "ProfileError", "Profile",
    "validate_mode", "validate_target", "validate_remote_name", "validate_ssh_host",
    "profile_id", "profile_root", "ensure_profile", "load_profile",
    "list_remotes", "get_remote", "save_remote", "remove_remote",
]
