"""ROS package manifests must be valid before anyone tries to build them.

Both manifests shipped with ``maintainer email="local@localhost"``, which
catkin_pkg rejects (reserved domain), so ``colcon build`` failed at configure
time. Nothing caught it because the manifests were never parsed by any test and
the workspace had never been built.
The structural checks run everywhere; the catkin_pkg check runs only where
catkin_pkg is installed (the ROS environment), so Windows core tests stay
dependency-free.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROS_SRC = Path(__file__).resolve().parents[1] / "ros2_ws" / "src"
MANIFESTS = sorted(ROS_SRC.glob("*/package.xml"))

#: Domains catkin_pkg refuses: they cannot receive mail and are treated as
#: placeholder noise rather than a real contact.
RESERVED_DOMAINS = ("localhost", "example.invalid", "invalid")


def test_manifests_are_present():
    names = {path.parent.name for path in MANIFESTS}
    assert names == {"piperlab_msgs", "piperlab_ros"}, f"unexpected manifests: {names}"


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_is_well_formed_and_has_required_tags(manifest: Path):
    root = ET.parse(manifest).getroot()
    assert root.tag == "package", "root element must be <package>"
    assert root.get("format") in {"2", "3"}, "package format must be 2 or 3"

    for tag in ("name", "version", "description", "maintainer", "license"):
        elements = root.findall(tag)
        assert len(elements) == 1, f"expected exactly one <{tag}>, found {len(elements)}"
        assert (elements[0].text or "").strip(), f"<{tag}> must not be empty"

    assert root.findtext("name") == manifest.parent.name, "package name must match directory"

    for maintainer in root.findall("maintainer"):
        email = maintainer.get("email") or ""
        assert email, "maintainer must declare an email"
        domain = email.rpartition("@")[2].lower()
        assert domain not in RESERVED_DOMAINS, (
            f"catkin_pkg rejects reserved domains such as {domain!r}; "
            "use a normal routable-looking address"
        )
        assert re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email), (
            f"maintainer email {email!r} is not a usable address"
        )


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_parses_with_catkin_pkg_when_available(manifest: Path):
    catkin_pkg = pytest.importorskip(
        "catkin_pkg.package", reason="catkin_pkg only exists in the ROS environment"
    )
    # catkin_pkg's InvalidPackage message can carry only the filename, so read
    # the exception's msg attribute as well when reporting a failure.
    try:
        catkin_pkg.parse_package_string(
            manifest.read_text(encoding="utf-8"), filename=str(manifest)
        )
    except catkin_pkg.InvalidPackage as exc:  # pragma: no cover - failure path
        pytest.fail(f"{manifest} is invalid: {exc.msg}")
