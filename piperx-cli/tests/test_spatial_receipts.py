"""Tests for spatial evidence receipts, PCA axis disambiguation, and candidate visual overlays."""
import base64
import copy
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from piperx_middleware import scene_geometry
from piperx_middleware.kinematics import PiperKinematics
from piperx_middleware.spatial_receipts import (
    OVERLAY_BANNER_LINE1,
    grasp_estimate,
    normalize_intrinsics,
    placement_geometry,
    preview_pose_receipt,
    project_base_to_pixel,
    relative_tcp_delta,
    render_candidate_overlay,
)
from piperx_middleware.task_plan import TASK_PLAN_TOOL, guide


def test_surface_summary_separates_minor_closing_axis_and_major_elongation_axis():
    rng = np.random.default_rng(42)
    # Elongated along base X (span ~0.10 m), narrow along base Y (width ~0.03 m)
    xs = np.linspace(0.25, 0.35, 50)
    ys = np.linspace(-0.015, 0.015, 20)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 0.04)])
    summary = scene_geometry._surface_summary(pts)

    assert not summary["orientation_ambiguous"]
    # Minor axis is along Y (+/-90 deg), major axis is along X (0 deg)
    assert abs(abs(summary["minor_axis_yaw_deg"]) - 90.0) < 1.0
    assert abs(summary["major_axis_yaw_deg"]) < 1.0
    assert summary["observed_minor_width_m"] == pytest.approx(0.03, abs=0.003)
    assert summary["observed_major_span_m"] == pytest.approx(0.10, abs=0.006)
    # Orthogonality check
    dot = float(np.dot(summary["minor_axis_xy"], summary["major_axis_xy"]))
    assert abs(dot) < 1e-6

    # Isotropic patch -> orientation_ambiguous is True
    iso_x, iso_y = np.meshgrid(np.linspace(0.28, 0.32, 25), np.linspace(-0.02, 0.02, 25))
    iso_pts = np.column_stack([iso_x.ravel(), iso_y.ravel(), np.full(iso_x.size, 0.04)])
    iso_summary = scene_geometry._surface_summary(iso_pts)
    assert iso_summary["orientation_ambiguous"] is True


def test_project_base_to_pixel_is_exact_inverse_of_deproject():
    intr = {"fx": 520.0, "fy": 520.0, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480}
    # Camera mounted at (0.3, 0.0, 0.8) looking downward with optical Z = -base Z
    base_from_camera = [
        [0.0, 1.0, 0.0, 0.30],
        [1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.80],
        [0.0, 0.0, 0.0, 1.00],
    ]
    us = np.array([180.0, 320.0, 460.0])
    vs = np.array([140.0, 240.0, 360.0])
    zs = np.array([0.65, 0.72, 0.58])
    pts_cam = scene_geometry._deproject(us, vs, zs, intr)
    pts_base = scene_geometry._transform_points(np.asarray(base_from_camera), pts_cam)

    projected = project_base_to_pixel(pts_base, intr, base_from_camera)
    assert len(projected) == 3
    for i, item in enumerate(projected):
        assert item["in_front"] is True
        assert item["in_image"] is True
        assert item["pixel_xy"] == pytest.approx([float(us[i]), float(vs[i])], abs=1e-6)
        assert item["depth_cam_m"] == pytest.approx(float(zs[i]), abs=1e-6)

    # Point behind camera (base Z = 1.2 > 0.8 -> z_cam = -0.4)
    behind = project_base_to_pixel([0.3, 0.0, 1.2], intr, base_from_camera)
    assert behind["in_front"] is False
    assert behind["pixel_xy"] is None


def test_grasp_and_placement_receipts_carry_relative_tcp_delta_and_disambiguated_axes():
    kin = PiperKinematics([0.0, 0.0, 0.1425])
    seed = [0.0, 20.0, -25.0, 0.0, 25.0, 0.0]
    tcp_now = kin.pose(seed)

    xs, ys = np.meshgrid(np.linspace(0.24, 0.32, 30), np.linspace(-0.015, 0.015, 15))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.14)])
    summary = scene_geometry._surface_summary(pts)

    grasp = grasp_estimate(
        kin, seed, summary, [0.28, 0.0], top=-0.14, height=0.06,
        view="live-0001", target_hash="abc123def456",
        measured_tcp_pose=tcp_now, sim_time_s=1.25, captured_at="1.25",
    )
    assert grasp["status"] == "ok"
    assert grasp["closing_axis_yaw_deg"] != grasp["object_major_axis_yaw_deg"]
    assert abs(abs(grasp["closing_axis_yaw_deg"]) - 90.0) < 1.0
    assert abs(grasp["object_major_axis_yaw_deg"]) < 1.0
    assert "NOT full 3D object bounding box" in grasp["observed_surface_spans_m"]["scope"]
    assert grasp["observed_surface_spans_m"]["visible_minor_span_m"] == pytest.approx(grasp["observed_width_m"])
    assert grasp["observed_surface_spans_m"]["visible_major_span_m"] == pytest.approx(summary["observed_major_span_m"])

    expected_delta = [
        grasp["poses"]["grasp"]["xyz_m"][i] - tcp_now["xyz_m"][i]
        for i in range(3)
    ]
    assert grasp["tcp_to_target_delta_m"] == pytest.approx(expected_delta)
    assert grasp["poses"]["grasp"]["tcp_to_target_delta_m"] == pytest.approx(expected_delta)
    assert grasp["relative_reference"]["frame"] == "base"
    assert grasp["relative_reference"]["observation_id"] == "live-0001"
    assert grasp["relative_reference"]["captured_sim_time_s"] == pytest.approx(1.25)
    assert grasp["relative_reference"]["measured_tcp_xyz_m"] == pytest.approx(tcp_now["xyz_m"])

    # Placement receipt check
    ref_summary = {
        "observed_bounds_m": {"lower": [0.24, -0.15, -0.20], "upper": [0.32, -0.09, -0.14]},
        "observed_extent_m": [0.08, 0.06, 0.06],
        "observed_minor_width_m": 0.06,
        "observed_major_span_m": 0.08,
    }
    plane = {"normal": [0.0, 0.0, 1.0], "point_on_plane_m": [0.0, 0.0, -0.20]}
    camera = {"base_from_camera": [[0.0, 1.0, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.8], [0.0, 0.0, 0.0, 1.0]]}
    place = placement_geometry(
        grasp, ref_summary, plane, camera, "left", 0.0, 0.02, 0.05,
        measured_tcp_pose=tcp_now, observation_id="live-0002", sim_time_s=2.5,
    )
    assert place["tcp_to_target_delta_m"] == pytest.approx(
        [place["xyz_m"][i] - tcp_now["xyz_m"][i] for i in range(3)]
    )
    assert place["relative_reference"]["observation_id"] == "live-0002"
    assert "reference_observed_surface_spans_m" in place

    # Preview receipt check
    prev = preview_pose_receipt(
        kin, seed, [0.28, 0.0, -0.05], [180.0, 0.0, 0.0],
        measured_tcp_pose=tcp_now, observation_id="live-0001", sim_time_s=1.25,
    )
    assert prev["status"] == "ok"
    assert prev["tcp_to_target_delta_m"] == pytest.approx(
        [0.28 - tcp_now["xyz_m"][0], 0.0 - tcp_now["xyz_m"][1], -0.05 - tcp_now["xyz_m"][2]]
    )


def test_render_candidate_overlay_preserves_raw_rgb_and_marks_candidate_only():
    raw_img = Image.new("RGB", (640, 480), color=(40, 50, 60))
    buf = io.BytesIO()
    raw_img.save(buf, format="JPEG", quality=92)
    raw_bytes = buf.getvalue()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest() if "hashlib" in globals() else None

    intr = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    base_from_camera = [
        [0.0, 1.0, 0.0, 0.30],
        [1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.80],
        [0.0, 0.0, 0.0, 1.00],
    ]
    jpeg_bytes, meta = render_candidate_overlay(
        raw_bytes,
        intrinsics=intr,
        base_from_camera=base_from_camera,
        calibration_id="calib-sha256-0001",
        observation_id="live-0001",
        sim_time_s=3.45,
        kind="grasp_candidate",
        target_xyz_m=[0.30, 0.00, 0.10],
        approach_xyz_m=[0.30, 0.00, 0.20],
        closing_yaw_deg=90.0,
        major_yaw_deg=0.0,
        opening_width_m=0.07,
        measured_tcp_xyz_m=[0.25, -0.05, 0.25],
        median_point_base_m=[0.30, 0.00, 0.10],
        observed_bounds_m={"lower": [0.26, -0.02, 0.07], "upper": [0.34, 0.02, 0.13]},
        object_roi=[260, 200, 380, 280],
        table_roi=[100, 100, 540, 420],
    )
    assert meta["status"] == "ok"
    assert meta["raw_image_preserved"] is True
    assert meta["banner"] == OVERLAY_BANNER_LINE1
    assert meta["certification"] == "candidate_only_not_collision_or_grasp_certified"
    assert meta["projected_pixels"]["target_pixel_xy"] == pytest.approx([320.0, 240.0], abs=1e-4)
    assert meta["projected_pixels"]["closing_finger_pixels_xy"] is not None
    assert meta["projected_pixels"]["major_axis_pixels_xy"] is not None

    decoded = Image.open(io.BytesIO(jpeg_bytes))
    assert decoded.size == (640, 480)
    assert jpeg_bytes != raw_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize('selection_kind', ['roi', 'polygon'])
async def test_online_loop_attaches_candidate_overlay_without_evicting_raw_live_frames(tmp_path, monkeypatch, selection_kind):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import run_longvil_online as online

    raw_img = Image.new("RGB", (640, 480), color=(30, 40, 50))
    buf = io.BytesIO()
    raw_img.save(buf, format="JPEG")
    valid_jpeg_b64 = base64.b64encode(buf.getvalue()).decode()

    intr = {"fx": 500.0, "fy": 500.0, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480}
    matrix = [[0.0, 1.0, 0.0, 0.30], [1.0, 0.0, 0.0, 0.00], [0.0, 0.0, -1.0, 0.80], [0.0, 0.0, 0.0, 1.00]]

    xs, ys = np.meshgrid(np.linspace(0.26, 0.34, 25), np.linspace(-0.015, 0.015, 15))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.14)])
    summary = scene_geometry._surface_summary(pts)

    class CalibratedSession:
        def __init__(self):
            self.time = 0.0
            self.moves = []
            self.geometry_calls = []
            self.plan = {"task_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "version": 1, "goal": "g", "interpretation": ""}

        async def list_tools(self):
            def t(name, schema):
                val = {"name": name, "description": name, "inputSchema": schema}
                return SimpleNamespace(model_dump=lambda **kw: val)
            return SimpleNamespace(tools=[
                t("move_to", {"type": "object", "required": ["request_id", "xyz_m"], "properties": {
                    "request_id": {"type": "string"}, "xyz_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}}, "additionalProperties": False}),
                t("camera_observe", {"type": "object"}),
                t("task_plan", TASK_PLAN_TOOL["inputSchema"]),
                t("robot_stop", {"type": "object", "properties": {"request_id": {"type": "string"}}}),
                t("scene_query", {"type": "object"}),
            ])

        async def call_tool(self, name, args):
            images = []
            if name == "task_plan":
                value = copy.deepcopy(self.plan)
            elif name == "camera_observe":
                self.time += 1.0
                frame = {
                    "source_stamp_s": self.time,
                    "sim_time_s": self.time,
                    "instance_id": "sim",
                    "connection_epoch": 1,
                    "calibration_id": "cal-001",
                    "intrinsics": intr,
                    "camera_info": {"intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], "base_from_camera": matrix},
                    "observation_path": f"/private/profile/observations/{self.time}/observation.json",
                    "robot_alignment": {"state": {"backend": "mujoco", "robot": {"q_deg": [0.0, 20.0, -25.0, 0.0, 25.0, 0.0], "gripper_width_m": 0.07}}},
                }
                value = {"simulated": True, "frames": [frame]}
                images = [SimpleNamespace(type="image", mimeType="image/jpeg", data=valid_jpeg_b64)]
            elif name == "scene_query":
                self.geometry_calls.append(copy.deepcopy(args))
                if args["operation"] == "fit_plane":
                    value = {"status": "ok", "plane_id": "plane-1", "normal": [0.0, 0.0, 1.0], "point_on_plane_m": [0.30, 0.0, -0.20]}
                else:
                    region = {k:v for k,v in args['observations'][0].items() if k in ('roi','polygon')}
                    value = {"status": "ok", "surface_summary": summary,
                             "observations":[{'region':region}], 'quality':{'voxels':len(pts)}}
            elif name == "move_to":
                self.time += 0.2
                self.moves.append(copy.deepcopy(args))
                value = {"status": "succeeded", "command": args, "request_id": args["request_id"]}
            else:
                raise AssertionError(name)
            return SimpleNamespace(isError=False, structuredContent=value, content=images)

    demo_img = tmp_path / "demo.jpg"
    demo_img.write_bytes(buf.getvalue())
    packet = {"source": {"sha256": "video"}, "frames": [{"frame_id": "f0", "timestamp_s": 0.0, "image_path": str(demo_img)}], "evaluation": {"scope": "recorded CV"}}
    args = SimpleNamespace(
        session_id="session", object_list=["red block"], max_rounds=4, max_seconds=60,
        completion_budget=65536, effort="low", max_tools=10, profile="private-profile", plan="off", skill="on",
    )
    evidence = {"source_sha256": "video", "initial": packet, "supplement": packet}
    session = CalibratedSession()
    payloads = []
    events = []
    object_region = [260, 200, 380, 280] if selection_kind=='roi' else [[260,200],[380,200],[320,280]]
    table_region = [100, 100, 540, 420] if selection_kind=='roi' else [[100,100],[200,100],[150,150]]

    async def paid(args, client, payload, ident):
        payloads.append(copy.deepcopy(payload))
        turn = len(payloads) - 1
        if turn == 0:
            call = {"id": "c0", "function": {"name": "grasp_target", "arguments": json.dumps({
                "observation_id": "live-0001", 'object_'+selection_kind:object_region, 'table_'+selection_kind:table_region
            })}}
        elif turn == 1:
            call = {"id": "c1", "function": {"name": "move_to", "arguments": json.dumps({
                "request_id": "motion-0001", "xyz_m": [0.30, 0.0, -0.08], "observation_id": "live-0001"
            })}}
        else:
            call = {"id": "c2", "function": {"name": "finish_task", "arguments": json.dumps({
                "summary": "done", "uncertainties": [], "observation_id": "live-0002"
            })}}
        return {"model": "deepseek-flash", "usage": {"completion_tokens": 60}, "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [call]}}]}

    async def remote(*vals):
        return {"sim_time_s": session.time, "bilateral_contact_objects": [], "objects": {}}

    monkeypatch.setattr(online, "paid", paid)
    monkeypatch.setattr(online, "remote", remote)
    report = {}
    await online.online_loop(session, args, evidence, events.append, tmp_path, report, None)

    assert report["status"] == "finished"
    grasp_evt = next(e for e in events if e["kind"] == "tool_result" and e["name"] == "grasp_target")
    res = grasp_evt["result"]
    assert res["status"] == "ok"
    assert "tcp_to_target_delta_m" in res
    assert res["relative_reference"]["observation_id"] == "live-0001"
    assert res["candidate_overlay"]["status"] == "ok"
    assert res["candidate_overlay"]["raw_image_preserved"] is True
    assert session.geometry_calls[0][selection_kind] == table_region
    assert session.geometry_calls[1]['observations'][0][selection_kind] == object_region
    assert res['selection_evidence']['target_region'] == {selection_kind:object_region}
    assert res['selection_evidence']['single_object_verified'] is False
    assert res['selection_evidence']['surface_quality']['voxels'] == len(pts)
    assert res['candidate_overlay']['selected_regions']['target'] == {selection_kind:object_region}

    # Raw live-0001.jpg is untouched, and a separate overlay file was saved
    assert (tmp_path / "live-0001.jpg").read_bytes() == buf.getvalue()
    overlay_files = list(tmp_path.glob("live-0001-grasp_candidate-*-overlay.jpg"))
    assert len(overlay_files) == 1

    # On turn 1 (right after grasp_target), model sees demo(1) + live-0001(1) + overlay(1) = 3 images
    turn1_msgs = payloads[1]["messages"]
    turn1_images = sum(p.get("type") == "image_url" for m in turn1_msgs if isinstance(m.get("content"), list) for p in m["content"])
    assert turn1_images == 3

    # On turn 2 (after move_to captured live-0002), stale pre-motion overlay image was evicted while live-0001 and live-0002 remain: demo(1) + live-0001(1) + live-0002(1) = 3 images
    turn2_msgs = payloads[2]["messages"]
    turn2_images = sum(p.get("type") == "image_url" for m in turn2_msgs if isinstance(m.get("content"), list) for p in m["content"])
    assert turn2_images == 3
    overlay_user_msgs = [m for m in turn2_msgs if m["role"] == "user" and isinstance(m.get("content"), list)
                         and any("CANDIDATE spatial overlay" in p.get("text", "") for p in m["content"] if p.get("type") == "text")]
    assert len(overlay_user_msgs) == 1
    assert not any(p.get("type") == "image_url" for p in overlay_user_msgs[0]["content"])


def test_top_surface_isolates_cylinder_from_side_walls_and_lower_neighbor_contamination():
    kin = PiperKinematics([0.0, 0.0, 0.1425])
    seed = [0.0, 20.0, -25.0, 0.0, 25.0, 0.0]
    # Red cylinder top disk at z = -0.10 m, centered at (0.35, -0.05), radius 0.024 m
    gx, gy = np.meshgrid(np.linspace(-0.024, 0.024, 15), np.linspace(-0.024, 0.024, 15))
    mask = (gx ** 2 + gy ** 2) <= 0.024 ** 2
    top_disk = np.column_stack([0.35 + gx[mask], -0.05 + gy[mask], np.full(int(mask.sum()), -0.10)])
    # Oblique side-wall points on the +Y side of the cylinder (y ~ -0.026 m, z in [-0.19, -0.115])
    theta = np.linspace(-0.9, 0.9, 20)
    zs = np.linspace(-0.19, -0.115, 12)
    tt, zz = np.meshgrid(theta, zs)
    side_wall = np.column_stack([
        0.35 + 0.024 * np.sin(tt.ravel()),
        -0.05 + 0.024 * np.cos(tt.ravel()),
        zz.ravel(),
    ])
    # Adjacent lower green cuboid at (0.42, -0.01), z = -0.17 m (contaminating the ROI)
    nx, ny = np.meshgrid(np.linspace(0.405, 0.435, 8), np.linspace(-0.025, 0.005, 8))
    neighbor = np.column_stack([nx.ravel(), ny.ravel(), np.full(nx.size, -0.17)])

    contaminated_pts = np.vstack([top_disk, side_wall, neighbor])
    summary = scene_geometry._surface_summary(contaminated_pts)

    assert summary["multiple_clusters_detected"] is True
    assert summary["cluster_count"] >= 2
    assert summary["dominant_cluster_fraction"] < 0.96
    assert summary["top_surface_summary"] is not None
    assert len(summary["sampled_points_base_m"]) <= 200

    # Contradictory top/full geometry cannot silently discard an adjacent object.
    with pytest.raises(ValueError, match="multiple visible clusters"):
        grasp_estimate(kin, seed, summary, summary["median_point_m"][:2],
                       top=-.10, height=.10, view="live-0001", target_hash="contaminated")
    clean_summary = scene_geometry._surface_summary(np.vstack([top_disk, side_wall]))
    # A clean cylinder keeps the top centre instead of drifting onto a side wall.
    grasp = grasp_estimate(
        kin, seed, clean_summary, clean_summary["median_point_m"][:2],
        top=-0.10, height=0.10, view="live-0001", target_hash="cyl123",
    )
    assert grasp["top_surface_used_for_grasp"] is True
    assert grasp["orientation_ambiguous"] is True
    assert grasp["closing_axis_yaw_deg"] == pytest.approx(0.0)
    assert grasp["poses"]["grasp"]["xyz_m"][0] == pytest.approx(0.35, abs=0.003)
    assert grasp["poses"]["grasp"]["xyz_m"][1] == pytest.approx(-0.05, abs=0.003)

    # Top placement onto the contaminated cylinder ROI also isolates the cylinder top face
    source = {
        "tcp_above_table_m": 0.0125,
        "observed_bounds_m": {"lower": [0.28, 0.08, -0.20], "upper": [0.32, 0.12, -0.15]},
        "poses": {"grasp": {"rpy_deg": [180.0, 0.0, 0.0], "xyz_m": [0.30, 0.10, -0.1875]}},
    }
    plane = {"normal": [0.0, 0.0, 1.0], "point_on_plane_m": [0.35, -0.05, -0.20]}
    place = placement_geometry(source, summary, plane, {}, "top", 0.0, 0.02)
    assert place["reference_top_surface_used"] is True
    assert place["reference_multiple_clusters_detected"] is True
    assert place["xyz_m"][0] == pytest.approx(0.35, abs=0.003)
    assert place["xyz_m"][1] == pytest.approx(-0.05, abs=0.003)


def test_elevated_object_grasp_uses_visible_thickness_instead_of_plunging_into_support():
    kin = PiperKinematics([0.0, 0.0, 0.1425])
    seed = [0.0, 20.0, -25.0, 0.0, 25.0, 0.0]
    # Object of thickness ~0.048 m (z in [-0.098, -0.050]) resting on a 0.10 m tall support above table z=-0.20 m
    xs, ys = np.meshgrid(np.linspace(0.30, 0.34, 12), np.linspace(-0.02, 0.02, 12))
    zs = np.linspace(-0.098, -0.050, xs.size)
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs])
    summary = scene_geometry._surface_summary(pts)

    grasp = grasp_estimate(
        kin, seed, summary, summary["median_point_m"][:2],
        top=-0.050, height=0.150, view="live-0001", target_hash="elevated123",
    )
    assert grasp["elevated_support_detected"] is True
    assert grasp["effective_object_height_m"] == pytest.approx(0.048, abs=0.003)
    assert "appears elevated above the tabletop" in grasp["assumption"]
    # Grasp Z stays in the upper object (~ -0.0865 m) rather than plunging to -0.1375 m
    assert grasp["poses"]["grasp"]["xyz_m"][2] > -0.095



@pytest.mark.parametrize('rotation', [0., 37., -61.])
def test_top_band_axis_is_checked_against_full_visible_extent(rotation):
    """Prism regression: a local 38mm strip must not prescribe a 118mm pinch."""
    class Kin:
        def solve(self, xyz, rpy, seed):
            return {'joints_deg':seed, 'position_error_m':0., 'orientation_error_deg':0.}
    summary = {
        'median_point_m':[.3,0.,-.17],
        'observed_bounds_m':{'lower':[.276,-.059,-.19],'upper':[.324,.059,-.15]},
        'observed_extent_m':[.048*abs(math.cos(math.radians(rotation)))+.118*abs(math.sin(math.radians(rotation))),
                             .048*abs(math.sin(math.radians(rotation)))+.118*abs(math.cos(math.radians(rotation))),.04],
        'minor_axis_yaw_deg':rotation, 'major_axis_yaw_deg':rotation+90,
        'observed_minor_width_m':.048, 'observed_major_span_m':.118,
        'orientation_ambiguous':False,
        'top_surface_summary':{
            'median_point_m':[.3,.01,-.156], 'point_count':377,
            'observed_extent_m':[.048,.038,.012],
            'observed_bounds_m':{'lower':[.276,0.,-.162],'upper':[.324,.038,-.15]},
            'minor_axis_yaw_deg':rotation+85.4, 'major_axis_yaw_deg':rotation-4.6,
            'observed_minor_width_m':.038, 'observed_major_span_m':.048,
            'orientation_ambiguous':False,
        },
    }
    value=grasp_estimate(Kin(),[0]*6,summary,[.3,0],-.15,.05,'live-1','test')
    assert value['closing_axis_yaw_deg']==pytest.approx(rotation)
    assert value['observed_width_m']==pytest.approx(.048)
    assert value['poses']['grasp']['xyz_m'][:2]==[.3,0.]
    assert value['geometry_consistency']['status']=='adjusted_to_visible_extent'
    assert value['geometry_consistency']['visible_span_at_original_yaw_m']>.118
    assert not value['top_surface_used_for_grasp'] and not value['collision_checked']
    # No orientation is feasible when even the full visible narrow axis is too wide.
    summary['observed_minor_width_m']=.08
    summary['observed_extent_m']=[.15,.15,.04]
    with pytest.raises(ValueError,match='gripper capability'):
        grasp_estimate(Kin(),[0]*6,summary,[.3,0],-.15,.05,'live-1','test')
