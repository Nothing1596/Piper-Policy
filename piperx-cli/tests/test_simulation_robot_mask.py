"""Robot self-occlusion depth filter for the MuJoCo sensor.

These tests use a fake renderer (no GL context) and exercise observe() mask
logic, failure handling and the geometry-worker contract. Native Linux
actual-render proof against a real mujoco.Renderer is done by the parent
integration, not here.
"""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import piperx_middleware.mujoco_backend as module
from piperx_middleware import scene_geometry

GEOM = 5  # mujoco.mjtObj.mjOBJ_GEOM
BODY = 1  # mujoco.mjtObj.mjOBJ_BODY

# Bodies: 0 world, 1 table, 2 cube, 3 base_link, 4 link1, 5 link2,
#         6 gripper_base, 7 gripper_link1
BODY_PARENTID = [0, 0, 0, 0, 3, 4, 5, 6]
BASE_LINK_ID = 3
ROBOT_BODIES = {3, 4, 5, 6, 7}
# Geoms: 0 floor (world), 1 table, 2 cube, 3..7 robot links/gripper
GEOM_BODYID = [0, 1, 2, 3, 4, 5, 6, 7]

HEIGHT, WIDTH = 6, 8
ROBOT_PIXELS = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]


def _seg_map():
    seg = np.full((HEIGHT, WIDTH, 2), -1, dtype=np.int32)  # background
    seg[0, 0] = (0, GEOM)    # floor (world body): kept
    seg[0, 1] = (1, GEOM)    # table: kept
    seg[0, 2] = (2, GEOM)    # task object: kept
    seg[1, 0] = (3, GEOM)    # base_link: masked
    seg[1, 1] = (4, GEOM)    # link1: masked
    seg[1, 2] = (5, GEOM)    # link2: masked
    seg[2, 0] = (6, GEOM)    # gripper_base: masked
    seg[2, 1] = (7, GEOM)    # gripper finger: masked
    seg[3, 0] = (3, BODY)    # non-geom entry on a robot body: kept
    seg[3, 1] = (-1, GEOM)   # unknown geom id: kept, must not index [-1]
    seg[3, 2] = (42, GEOM)   # out-of-range geom id: kept, must not crash
    return seg


def _images():
    rgb = np.arange(HEIGHT * WIDTH * 3, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    depth = (1.0 + np.arange(HEIGHT * WIDTH, dtype=np.float32) / 100.0).reshape(HEIGHT, WIDTH)
    return rgb, depth, _seg_map()


class FakeRenderer:
    """Mimics mujoco.Renderer mode flags without opening a graphics context."""

    def __init__(self, rgb, depth, seg, *, depth_failure=False, seg_failure=False):
        self.rgb = rgb
        self.depth_image = depth
        self.seg = seg
        self.depth_failure = depth_failure
        self.seg_failure = seg_failure
        self.depth_mode = False
        self.seg_mode = False
        self.rgb_renders = 0
        self.depth_renders = 0
        self.seg_renders = 0

    def update_scene(self, data, *, camera, scene_option):
        assert camera == 'overview_camera'
        assert not scene_option.sitegroup.any(), 'debug sites must stay excluded'
        assert scene_option.geomgroup.all(), 'physical geometry stays visible'

    def render(self):
        if self.seg_mode:
            self.seg_renders += 1
            if self.seg_failure:
                raise RuntimeError('test segmentation failure')
            return self.seg.copy()
        if self.depth_mode:
            self.depth_renders += 1
            if self.depth_failure:
                raise RuntimeError('test depth failure')
            return self.depth_image.copy()
        self.rgb_renders += 1
        return self.rgb

    def enable_depth_rendering(self):
        self.depth_mode = True
        self.seg_mode = False

    def disable_depth_rendering(self):
        self.depth_mode = False

    def enable_segmentation_rendering(self):
        self.seg_mode = True
        self.depth_mode = False

    def disable_segmentation_rendering(self):
        self.seg_mode = False


def _backend(monkeypatch, renderer, *, topology='ok'):
    options = SimpleNamespace(sitegroup=np.ones(6, dtype=int), geomgroup=np.ones(6, dtype=int))

    def name2id(model, objtype, name):
        assert objtype == BODY
        return -1 if topology == 'missing_base' else BASE_LINK_ID

    fake_mujoco = SimpleNamespace(MjvOption=lambda: options,
                                  mjtObj=SimpleNamespace(mjOBJ_BODY=BODY, mjOBJ_GEOM=GEOM),
                                  mj_name2id=name2id)
    if topology == 'no_name2id':
        fake_mujoco = SimpleNamespace(MjvOption=lambda: options)
    monkeypatch.setattr(module, 'mujoco', fake_mujoco)

    backend = module.MujocoBackend.__new__(module.MujocoBackend)
    backend._thread_local = SimpleNamespace(renderer=renderer, renderer_dims=(640, 480))
    backend.lock = threading.RLock()
    backend.render_lock = threading.RLock()
    backend.model = SimpleNamespace(cam_fovy=[55.0])
    if topology != 'no_arrays':
        backend.model.body_parentid = np.array(BODY_PARENTID)
        backend.model.geom_bodyid = np.array(GEOM_BODYID)
    backend.camera_id = 0
    backend.epoch = 1
    backend.data = SimpleNamespace(time=5.0,
                                   cam_xpos=np.array([[0.3, 0.0, 0.9]]),
                                   cam_xmat=np.eye(3).reshape(1, 9))
    backend.snapshot = lambda: SimpleNamespace(public=lambda: {'q_deg': [0.0] * 6})
    return backend


def test_robot_subtree_pixels_become_unavailable_depth(monkeypatch):
    rgb, depth, seg = _images()
    renderer = FakeRenderer(rgb, depth, seg)
    backend = _backend(monkeypatch, renderer)
    original = depth.copy()

    result = backend.observe()

    assert result['rgb'] is rgb  # RGB passes through exactly untouched
    out = result['depth']
    assert out.shape == (HEIGHT, WIDTH) and out.dtype == np.float32
    for pixel in ROBOT_PIXELS:
        assert np.isnan(out[pixel]), f'robot pixel {pixel} must be unavailable'
    assert int(np.isnan(out).sum()) == len(ROBOT_PIXELS)
    kept = np.ones((HEIGHT, WIDTH), dtype=bool)
    kept[tuple(zip(*ROBOT_PIXELS))] = False
    assert np.array_equal(out[kept], original[kept]), \
        'object/table/floor/background/ambiguous pixels keep their measured depth'

    meta = result['depth_filter']
    assert meta == {'applied': True,
                    'method': 'capture_aligned_geom_segmentation_robot_subtree',
                    'filtered_pixels': len(ROBOT_PIXELS),
                    'image_pixels': HEIGHT * WIDTH,
                    'robot_body_count': len(ROBOT_BODIES)}
    json.dumps(meta)  # provenance/count metadata must survive JSON transport
    assert 'segmentation' not in result and 'seg' not in result
    assert renderer.depth_mode is False and renderer.seg_mode is False


def test_masked_depth_never_reveals_hidden_surface(monkeypatch):
    rgb, depth, seg = _images()
    depth[2, 1] = 0.777  # measured depth at the gripper pixel
    depth[0, 2] = 0.555  # task object pixel beside the gripper
    renderer = FakeRenderer(rgb, depth, seg)
    backend = _backend(monkeypatch, renderer)

    out = backend.observe()['depth']

    assert np.isnan(out[2, 1]), 'gripper pixel becomes unavailable, not hidden-object depth'
    assert out[0, 2] == pytest.approx(0.555), 'visible object pixel keeps its own measured depth'
    finite = out[np.isfinite(out)]
    assert not np.isclose(finite, 0.777).any(), 'masked depth must not migrate anywhere'
    assert np.isfinite(depth).all(), 'renderer buffer itself was not mutated in place'


@pytest.mark.parametrize('topology', ['missing_base', 'no_arrays', 'no_name2id'])
def test_missing_topology_leaves_depth_unfiltered(monkeypatch, topology):
    rgb, depth, seg = _images()
    renderer = FakeRenderer(rgb, depth, seg)
    backend = _backend(monkeypatch, renderer, topology=topology)

    result = backend.observe()

    assert np.array_equal(result['depth'], depth)
    assert not np.isnan(result['depth']).any()
    meta = result['depth_filter']
    assert meta['applied'] is False
    assert meta['reason'] == 'robot_topology_unavailable'
    assert meta['filtered_pixels'] == 0 and meta['robot_body_count'] == 0
    assert renderer.seg_renders == 0, 'no segmentation pass without robot topology'
    assert renderer.depth_mode is False and renderer.seg_mode is False


@pytest.mark.parametrize('stage', ['depth', 'segmentation'])
def test_renderer_failure_restores_both_modes(monkeypatch, stage):
    rgb, depth, seg = _images()
    renderer = FakeRenderer(rgb, depth, seg,
                            depth_failure=stage == 'depth',
                            seg_failure=stage == 'segmentation')
    backend = _backend(monkeypatch, renderer)

    with pytest.raises(RuntimeError, match=f'test {stage} failure'):
        backend.observe()

    assert renderer.depth_mode is False and renderer.seg_mode is False
    assert renderer.rgb_renders == 1 and renderer.depth_renders == 1
    assert renderer.seg_renders == (1 if stage == 'segmentation' else 0)


def test_malformed_segmentation_raises_after_mode_restore(monkeypatch):
    rgb, depth, _ = _images()
    renderer = FakeRenderer(rgb, depth, np.zeros((HEIGHT, WIDTH), np.int32))  # no (objid, objtype)
    backend = _backend(monkeypatch, renderer)

    with pytest.raises(RuntimeError, match='Segmentation render shape'):
        backend.observe()

    assert renderer.depth_mode is False and renderer.seg_mode is False


def test_repeated_captures_reuse_renderer_and_restore_modes(monkeypatch):
    rgb, depth, seg = _images()
    renderer = FakeRenderer(rgb, depth, seg)
    backend = _backend(monkeypatch, renderer)
    pristine = depth.copy()

    first = backend.observe()
    second = backend.observe()

    assert backend._thread_local.renderer is renderer, 'per-thread renderer is reused'
    for result in (first, second):
        assert result['depth_filter']['applied'] is True
        assert result['depth_filter']['filtered_pixels'] == len(ROBOT_PIXELS)
        assert int(np.isnan(result['depth']).sum()) == len(ROBOT_PIXELS)
        assert result['rgb'] is rgb
    assert np.array_equal(first['depth'], second['depth'], equal_nan=True)
    assert renderer.depth_mode is False and renderer.seg_mode is False
    assert np.array_equal(renderer.depth_image, pristine), 'source depth never mutated in place'
    assert renderer.rgb_renders == 2 and renderer.depth_renders == 2 and renderer.seg_renders == 2


def test_geometry_reads_masked_depth_unavailable_and_finite_depth_measured(monkeypatch, tmp_path):
    height = width = 16
    block = np.s_[9:13, 9:13]  # robot footprint in the capture
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1.0, dtype=np.float32)
    seg = np.full((height, width, 2), -1, dtype=np.int32)
    seg[..., 0] = 1            # whole frame shows the table geom
    seg[..., 1] = GEOM
    seg[block] = (4, GEOM)     # robot link geom covers the block
    renderer = FakeRenderer(rgb, depth, seg)
    backend = _backend(monkeypatch, renderer)

    observation = backend.observe()
    assert np.isnan(observation['depth'][block]).all()

    task = tmp_path / 'tasks' / 't1'
    task.mkdir(parents=True)
    (task / 'plan.json').write_text(json.dumps({'task_id': 't1', 'steps': []}), encoding='utf-8')
    folder = tmp_path / 'observations' / 'sim1'
    folder.mkdir(parents=True)
    depth_file = folder / 'depth.npy'
    np.save(depth_file, observation['depth'])
    doc = {
        'camera_serial': 'sim-1',
        'captured_at': '2026-01-01T00:00:00+00:00',
        'frame': 'camera_color_optical',
        'depth_path': str(depth_file),
        'depth_scale_m': 1.0,
        'provenance': 'mujoco_sensor',
        'intrinsics': {'width': width, 'height': height,
                       'fx': 50.0, 'fy': 50.0, 'ppx': 8.0, 'ppy': 8.0,
                       'coeffs': [0.0] * 5},
    }
    observation_path = folder / 'observation.json'
    observation_path.write_text(json.dumps(doc), encoding='utf-8')

    hidden = scene_geometry.compute(tmp_path, {'operation': 'surface_point', 'task_id': 't1',
                                               'observation_path': str(observation_path),
                                               'pixel': [11, 11], 'patch_half_size': 1})
    assert hidden['status'] == 'unavailable'
    assert 'missing or discontinuous' in hidden['reason']

    visible = scene_geometry.compute(tmp_path, {'operation': 'surface_point', 'task_id': 't1',
                                                'observation_path': str(observation_path),
                                                'pixel': [3, 3], 'patch_half_size': 1})
    assert visible['status'] == 'ok'
    assert visible['point_camera_m'] == pytest.approx([-0.1, -0.1, 1.0])
    assert visible['source']['simulated'] is True
