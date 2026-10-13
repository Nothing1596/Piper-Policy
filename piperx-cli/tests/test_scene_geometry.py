"""Tests for the local CPU scene-geometry worker (piperx_middleware.scene_geometry)."""
import hashlib
import json
import os

import numpy as np
import pytest

from piperx_middleware import scene_geometry

HEIGHT, WIDTH = 48, 64
PPX, PPY = 32.0, 24.0


def _task(root, task_id='t1'):
    task = root / 'tasks' / task_id
    task.mkdir(parents=True, exist_ok=True)
    (task / 'plan.json').write_text(json.dumps({'task_id': task_id, 'steps': []}), encoding='utf-8')
    return task


def _observation(root, name, depth, *, fx=50.0, fy=50.0, ppx=PPX, ppy=PPY, scale=0.001,
                 serial='cam-1', frame='camera_color_optical', depth_path=None,
                 calibration_id=None, coeffs=None, width=None, height=None):
    directory = root / 'observations' / name
    directory.mkdir(parents=True, exist_ok=True)
    depth_file = directory / 'depth.npy'
    np.save(depth_file, depth)
    doc = {
        'camera_serial': serial, 'captured_at': '2026-01-01T00:00:00+00:00', 'frame': frame,
        'depth_path': str(depth_file) if depth_path is None else str(depth_path),
        'depth_scale_m': scale,
        'intrinsics': {'width': depth.shape[1] if width is None else width,
                       'height': depth.shape[0] if height is None else height,
                       'fx': fx, 'fy': fy, 'ppx': ppx, 'ppy': ppy,
                       'coeffs': [0.0] * 5 if coeffs is None else coeffs},
    }
    if calibration_id is not None:
        doc['calibration_id'] = calibration_id
    path = directory / 'observation.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    return str(path)


def _transform(rotation=None, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix.tolist()


def _flat(depth_mm=1000):
    return np.full((HEIGHT, WIDTH), depth_mm, dtype=np.uint16)


def _request(operation, **fields):
    return {'operation': operation, 'task_id': 't1', **fields}


def test_model_polygon_separates_two_surfaces_without_automatic_segmentation(tmp_path):
    """Actual worker replay: a box merges both blobs, a selected polygon does not."""
    _task(tmp_path)
    depth = _flat()
    depth[16:32, 8:24] = 950
    depth[16:32, 36:52] = 950
    observation = _observation(tmp_path, 'two-surfaces', depth, fx=500, fy=500)
    transform = _transform(rotation=np.diag([1.,-1.,-1.]), translation=(0,0,1))
    shared = {'observation_path':observation,'base_from_camera':transform,'calibration_id':'test'}
    plane = scene_geometry.compute(tmp_path, _request('fit_plane', **shared,
        roi=[0,0,64,12], reference_normal=[0,0,1],ransac={'min_width_m':.005}))
    assert plane['status']=='ok'
    results={}
    for name,region in [('mixed',{'roi':[6,14,54,34]}),
                        ('selected',{'polygon':[[6,14],[25,14],[25,34],[6,34]]})]:
        results[name]=scene_geometry.compute(tmp_path,_request('accumulate_region',
            target_id='target',batch_id=name,plane_id=plane['plane_id'],
            observations=[dict(shared,**region)]))
        assert results[name]['status']=='ok'
    mixed=results['mixed']['surface_summary']['observed_bounds_m']
    selected=results['selected']['surface_summary']['observed_bounds_m']
    assert mixed['lower'][0] < -.01 and mixed['upper'][0] > .01
    assert selected['upper'][0] < -.01
    assert selected['upper'][0]-selected['lower'][0] < .035
    # No semantic identity or private object ID was given to the worker.
    assert results['selected']['observations'][0]['region']['polygon']==[[6.,14.],[25.,14.],[25.,34.],[6.,34.]]


# --------------------------------------------------------------------------- #
# known geometry / transforms
# --------------------------------------------------------------------------- #
def test_surface_point_camera_frame_and_persisted_result(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat())
    result = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=observation, pixel=[32, 24]))
    assert result['status'] == 'ok'
    assert result['point_camera_m'] == pytest.approx([0.0, 0.0, 1.0])
    assert result['point_m'] == result['point_camera_m']
    assert result['frame'] == 'camera_color_optical'
    assert result['units'] == 'm'
    assert result['transform_applied'] is False
    assert result['quality']['valid_fraction'] == 1.0
    assert result['quality']['patch_pixels'] == 25
    stored = tmp_path / result['result_path']
    assert stored.is_file()
    assert json.loads(stored.read_text(encoding='utf-8'))['result_id'] == result['result_id']
    assert 'no grasp' in result['safety_note']


def test_surface_point_known_rotation_and_translation(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat())
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=observation, pixel=[42, 24],
        base_from_camera=_transform(rotation, (0.1, 0.2, 0.3)), calibration_id='calib-1'))
    assert result['status'] == 'ok'
    assert result['point_camera_m'] == pytest.approx([0.2, 0.0, 1.0])
    assert result['point_base_m'] == pytest.approx([0.1, 0.4, 1.3])
    assert result['point_m'] == result['point_base_m']
    assert result['frame'] == 'base'
    assert result['transform_applied'] is True


def test_surface_point_rejects_missing_and_discontinuous_depth(tmp_path):
    _task(tmp_path)
    holed = _flat()
    holed[24, 32] = 0
    hole = _observation(tmp_path, 'hole', holed)
    result = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=hole, pixel=[32, 24]))
    assert result['status'] == 'unavailable'
    assert 'missing or discontinuous' in result['reason']
    assert result['operation'] == 'surface_point' and result['task_id'] == 't1'

    stepped = _flat()
    stepped[:, 32:] = 1200
    step = _observation(tmp_path, 'step', stepped)
    result = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=step, pixel=[32, 24]))
    assert result['status'] == 'unavailable'
    assert 'missing or discontinuous' in result['reason']


def test_nonzero_distortion_and_intrinsics_mismatch_are_unavailable(tmp_path):
    _task(tmp_path)
    distorted = _observation(tmp_path, 'dist', _flat(), coeffs=[0.1, 0.0, 0.0, 0.0, 0.0])
    result = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=distorted, pixel=[32, 24]))
    assert result['status'] == 'unavailable'
    assert 'distortion' in result['reason']

    mismatched = _observation(tmp_path, 'shape', _flat(), width=WIDTH // 2)
    result = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=mismatched, pixel=[32, 24]))
    assert result['status'] == 'unavailable'
    assert 'does not match intrinsics' in result['reason']


# --------------------------------------------------------------------------- #
# request validation: task, calibration, paths
# --------------------------------------------------------------------------- #
def test_task_plan_and_safe_ids_are_required(tmp_path):
    observation = _observation(tmp_path, 'o1', _flat())
    with pytest.raises(ValueError, match='plan.json'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', task_id='missing', observation_path=observation, pixel=[32, 24]))
    with pytest.raises(ValueError, match='safe id'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', task_id='../evil', observation_path=observation, pixel=[32, 24]))
    with pytest.raises(ValueError, match='operation must be one of'):
        scene_geometry.compute(tmp_path, _request(
            'explode', observation_path=observation, pixel=[32, 24]))


def test_base_transform_requires_matching_calibration(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    transform = _transform(translation=(0.1, 0.2, 0.3))
    with pytest.raises(ValueError, match='calibration_id is required'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path=observation, pixel=[32, 24],
            base_from_camera=transform))
    with pytest.raises(ValueError, match='conflicts with the calibration_id'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path=observation, pixel=[32, 24],
            base_from_camera=transform, calibration_id='calib-2'))
    with pytest.raises(ValueError, match='orthonormal'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path=observation, pixel=[32, 24],
            base_from_camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]],
            calibration_id='calib-1'))


def test_path_traversal_and_symlinks_are_rejected(tmp_path):
    _task(tmp_path)
    (tmp_path / 'observations').mkdir(exist_ok=True)
    outside = tmp_path.parent / 'outside-observation.json'
    outside.write_text('{"intrinsics": {}}', encoding='utf-8')
    with pytest.raises(ValueError, match='stay under'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path='../outside-observation.json', pixel=[0, 0]))
    with pytest.raises(ValueError, match='stay under'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path=str(outside), pixel=[0, 0]))
    with pytest.raises(ValueError, match='does not exist'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path='observations/nope.json', pixel=[0, 0]))

    link = tmp_path / 'observations' / 'inside-link.json'
    _observation(tmp_path, 'o1', _flat())
    try:
        os.symlink(tmp_path / 'observations' / 'o1' / 'observation.json', link)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip('symlinks unavailable')
    with pytest.raises(ValueError, match='symlink'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path='observations/inside-link.json', pixel=[0, 0]))
    outside_link = tmp_path / 'observations' / 'outside-link.json'
    os.symlink(outside, outside_link)
    with pytest.raises(ValueError, match='stay under'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path='observations/outside-link.json', pixel=[0, 0]))


def test_depth_path_must_stay_inside_root(tmp_path):
    _task(tmp_path)
    outside_depth = tmp_path.parent / 'outside-depth.npy'
    np.save(outside_depth, _flat())
    observation = _observation(tmp_path, 'o1', _flat(), depth_path=outside_depth)
    with pytest.raises(ValueError, match='stay under'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path=observation, pixel=[32, 24]))
    traversal = _observation(tmp_path, 'o2', _flat(), depth_path='../../../outside-depth.npy')
    with pytest.raises(ValueError, match='stay under'):
        scene_geometry.compute(tmp_path, _request(
            'surface_point', observation_path=traversal, pixel=[32, 24]))


# --------------------------------------------------------------------------- #
# accumulate_region
# --------------------------------------------------------------------------- #
def _accumulate_request(observation, **overrides):
    entry = {'observation_path': observation, 'roi': [10, 10, 30, 20],
             'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
             'calibration_id': 'calib-1'}
    request = {'operation': 'accumulate_region', 'task_id': 't1', 'target_id': 'table',
               'batch_id': 'b1', 'voxel_size_m': 0.02, 'stride': 2, 'observations': [entry]}
    request.update(overrides)
    return request


def test_accumulate_region_transform_dedup_and_persistence(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    request = _accumulate_request(observation)
    result = scene_geometry.compute(tmp_path, request)
    assert result['status'] == 'ok'
    assert result['duplicates_skipped'] == 0
    assert result['frame'] == 'base' and result['units'] == 'm'
    summary=result['surface_summary']
    assert summary['median_point_m'][2]==pytest.approx(1.3)
    assert summary['observed_extent_m'][2]==pytest.approx(0.)
    assert 0 < summary['observed_minor_width_m'] < .3
    assert 'Visible ROI surface' in summary['scope']
    points = np.load(tmp_path / result['points_npz'])['points_base_m']
    entry_index = np.load(tmp_path / result['points_npz'])['entry_index']
    assert len(points) == len(entry_index) == result['points'] > 0
    assert np.allclose(points[:, 2], 1.3)
    assert points[:, 0].min() == pytest.approx((10 - PPX) / 50.0 + 0.1, abs=0.02)
    assert points[:, 0].max() == pytest.approx((28 - PPX) / 50.0 + 0.1, abs=0.02)
    keys = np.floor(points / 0.02).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == len(points)
    metadata = json.loads((tmp_path / result['metadata_json']).read_text(encoding='utf-8'))
    assert metadata['result_id'] == 'batch.table.b1'
    assert metadata['point_count'] == result['points']
    assert 'no free-space' in metadata['notes'][0]
    assert 'no grasp' in metadata['safety_note']
    assert not list((tmp_path / 'tasks' / 't1').rglob('*.tmp'))

    duplicated = dict(request, observations=[request['observations'][0]] * 2)
    repeat = scene_geometry.compute(tmp_path, duplicated)
    assert repeat['status'] == 'ok'
    assert repeat['duplicates_skipped'] == 1
    assert repeat['points'] == result['points']
    assert np.array_equal(points, np.load(tmp_path / repeat['points_npz'])['points_base_m'])

    again = scene_geometry.compute(tmp_path, request)
    assert again['points'] == result['points']
    assert np.array_equal(points, np.load(tmp_path / again['points_npz'])['points_base_m'])


def test_accumulate_region_rejects_conflicts_and_missing_calibration(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), serial='cam-1', calibration_id='calib-1')
    request = _accumulate_request(observation)
    moved = dict(request['observations'][0])
    moved['base_from_camera'] = _transform(translation=(0.9, 0.0, 0.0))
    with pytest.raises(ValueError, match='conflicting geometry'):
        scene_geometry.compute(tmp_path, dict(request, observations=[request['observations'][0],
                                                                     moved]))
    second = _observation(tmp_path, 'o2', _flat(1001), serial='cam-1', calibration_id='calib-2')
    mixed = dict(request['observations'][0], observation_path=second, calibration_id='calib-2')
    with pytest.raises(ValueError, match='mixed calibration'):
        scene_geometry.compute(tmp_path, dict(request, observations=[request['observations'][0],
                                                                     mixed]))
    with pytest.raises(ValueError, match='base_from_camera and calibration_id are required'):
        scene_geometry.compute(tmp_path, dict(request, observations=[
            {'observation_path': observation, 'roi': [10, 10, 30, 20]}]))
    with pytest.raises(ValueError, match='exactly one of roi or polygon'):
        scene_geometry.compute(tmp_path, dict(request, observations=[
            {'observation_path': observation, 'roi': [10, 10, 30, 20],
             'polygon': [[1, 1], [2, 2], [3, 3]], 'base_from_camera':
                 request['observations'][0]['base_from_camera'], 'calibration_id': 'calib-1'}]))


def test_identical_depth_bytes_from_different_observations_are_distinct_views(tmp_path):
    """Different cameras/poses can emit identical depth arrays; identity is the path."""
    _task(tmp_path)
    first = _observation(tmp_path, 'cam-a', _flat(), serial='cam-1', calibration_id='calib-1')
    second = _observation(tmp_path, 'cam-b', _flat(), serial='cam-2', calibration_id='calib-2')
    result = scene_geometry.compute(tmp_path, {
        'operation': 'accumulate_region', 'task_id': 't1', 'target_id': 'table',
        'batch_id': 'two-cameras', 'voxel_size_m': 0.02, 'stride': 2,
        'observations': [
            {'observation_path': first, 'roi': [10, 10, 30, 20],
             'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
             'calibration_id': 'calib-1'},
            {'observation_path': second, 'roi': [10, 10, 30, 20],
             'base_from_camera': _transform(translation=(0.5, 0.2, 0.3)),
             'calibration_id': 'calib-2'}]})
    assert result['status'] == 'ok'
    assert result['duplicates_skipped'] == 0
    assert result['quality']['source_observation_count'] == 2
    records = result['observations']
    assert {record['source']['camera_serial'] for record in records} == {'cam-1', 'cam-2'}
    # byte-identical depths, yet two distinct observations with their own calibration
    assert len({record['source']['depth_digest'] for record in records}) == 1
    assert {record['calibration_id'] for record in records} == {'calib-1', 'calib-2'}


def test_batch_ids_are_immutable_and_torn_pairs_are_rejected(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    request = _accumulate_request(observation)
    first = scene_geometry.compute(tmp_path, request)
    assert first['status'] == 'ok'
    npz_path = tmp_path / first['points_npz']
    meta_path = tmp_path / first['metadata_json']
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    assert metadata['npz_sha256'] == hashlib.sha256(npz_path.read_bytes()).hexdigest()

    repeat = scene_geometry.compute(tmp_path, request)
    assert repeat['result_id'] == first['result_id']
    assert repeat['points'] == first['points']

    conflicting = dict(request, observations=[dict(request['observations'][0],
                                                    roi=[12, 12, 30, 20])])
    with pytest.raises(ValueError, match='immutable'):
        scene_geometry.compute(tmp_path, conflicting)
    assert json.loads(meta_path.read_text(encoding='utf-8')) == metadata
    assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == metadata['npz_sha256']

    # A torn pair: the stored NPZ no longer matches the digest committed in metadata.
    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays['points_base_m'] = arrays['points_base_m'] + 1.0
    np.savez(npz_path, **arrays)
    fit_request = {'operation': 'fit_plane', 'task_id': 't1', 'target_id': 'table',
                   'batch_id': 'b1'}
    with pytest.raises(ValueError, match='digest mismatch'):
        scene_geometry.compute(tmp_path, fit_request)

    # A missing metadata half is equally invalid, even though the NPZ is present.
    meta_path.unlink()
    with pytest.raises(ValueError, match='does not exist'):
        scene_geometry.compute(tmp_path, fit_request)

    # The orphan NPZ cannot be silently overwritten either; only the operator may repair it.
    with pytest.raises(ValueError, match='torn pair'):
        scene_geometry.compute(tmp_path, request)
    npz_path.unlink()
    repaired = scene_geometry.compute(tmp_path, request)
    assert repaired['status'] == 'ok' and repaired['result_id'] == first['result_id']


def test_accumulate_region_polygon_and_plane_fit_from_batch(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    entry = {'observation_path': observation, 'polygon': [[10, 10], [40, 10], [40, 30], [10, 30]],
             'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
             'calibration_id': 'calib-1'}
    batch = scene_geometry.compute(tmp_path, {
        'operation': 'accumulate_region', 'task_id': 't1', 'target_id': 'table',
        'batch_id': 'poly', 'voxel_size_m': 0.02, 'stride': 2, 'observations': [entry]})
    assert batch['status'] == 'ok' and batch['points'] > 0
    points = np.load(tmp_path / batch['points_npz'])['points_base_m']
    assert points[:, 0].min() >= (10 - PPX) / 50.0 + 0.1 - 0.03
    assert points[:, 0].max() <= (39 - PPX) / 50.0 + 0.1 + 0.03
    fit = scene_geometry.compute(tmp_path, {
        'operation': 'fit_plane', 'task_id': 't1', 'target_id': 'table', 'batch_id': 'poly'})
    assert fit['status'] == 'ok'
    assert fit['normal'] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert fit['residual_m']['rms'] <= 1e-9
    # a batch-derived plane inherits the committed batch calibration, not None
    assert fit['calibration_id'] == 'calib-1'
    point = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=observation, pixel=[20, 15],
        base_from_camera=_transform(translation=(0.1, 0.2, 0.3)), calibration_id='calib-1'))
    query = scene_geometry.compute(tmp_path, {
        'operation': 'point_plane_distance', 'task_id': 't1',
        'point_id': point['result_id'], 'plane_id': fit['plane_id']})
    assert query['status'] == 'ok'
    assert query['signed_distance_m'] == pytest.approx(0.0, abs=1e-9)
    assert query['calibration_match'] is True
    assert query['clearance_verified'] is False
    with pytest.raises(ValueError, match='does not exist'):
        scene_geometry.compute(tmp_path, {
            'operation': 'fit_plane', 'task_id': 't1', 'target_id': 'table',
            'batch_id': 'unknown'})


def test_batch_plane_keeps_per_serial_calibration_provenance_and_sources(tmp_path):
    _task(tmp_path)
    first = _observation(tmp_path, 'cam-a', _flat(), serial='cam-1', calibration_id='calib-1')
    entry = {'observation_path': first, 'roi': [10, 10, 30, 20],
             'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
             'calibration_id': 'calib-1'}
    batch = scene_geometry.compute(tmp_path, {
        'operation': 'accumulate_region', 'task_id': 't1', 'target_id': 'table',
        'batch_id': 'one', 'voxel_size_m': 0.02, 'stride': 2, 'observations': [entry]})
    assert batch['status'] == 'ok'
    fit = scene_geometry.compute(tmp_path, {
        'operation': 'fit_plane', 'task_id': 't1', 'target_id': 'table', 'batch_id': 'one'})
    assert fit['status'] == 'ok'
    assert fit['calibration_id'] == 'calib-1'
    assert fit['calibration_provenance'] == [{
        'camera_serial': 'cam-1', 'calibration_id': 'calib-1',
        'observation': batch['observations'][0]['source']['observation'],
        'depth_digest': batch['observations'][0]['source']['depth_digest']}]
    assert fit['source']['points']['batch_metadata'] == batch['metadata_json']
    assert fit['source']['points']['observations'] == batch['observations']
    stored = json.loads((tmp_path / fit['result_path']).read_text(encoding='utf-8'))
    assert stored['calibration_provenance'] == fit['calibration_provenance']

    # Two cameras in one batch: no single top-level calibration, but both mappings survive.
    second = _observation(tmp_path, 'cam-b', _flat(), serial='cam-2', calibration_id='calib-2')
    other = dict(entry, observation_path=second,
                 base_from_camera=_transform(translation=(0.6, 0.2, 0.3)),
                 calibration_id='calib-2')
    mixed = scene_geometry.compute(tmp_path, {
        'operation': 'accumulate_region', 'task_id': 't1', 'target_id': 'table',
        'batch_id': 'two', 'voxel_size_m': 0.02, 'stride': 2, 'observations': [entry, other]})
    assert mixed['status'] == 'ok'
    fit_mixed = scene_geometry.compute(tmp_path, {
        'operation': 'fit_plane', 'task_id': 't1', 'target_id': 'table', 'batch_id': 'two'})
    assert fit_mixed['status'] == 'ok'
    assert fit_mixed['calibration_id'] is None
    assert {(record['camera_serial'], record['calibration_id'])
            for record in fit_mixed['calibration_provenance']} == {('cam-1', 'calib-1'),
                                                                   ('cam-2', 'calib-2')}
    assert len(fit_mixed['source']['points']['observations']) == 2

    # The plane for a shared camera serial still verifies against that camera's calibration.
    point = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=first, pixel=[20, 15],
        base_from_camera=_transform(translation=(0.1, 0.2, 0.3)), calibration_id='calib-1'))
    query = scene_geometry.compute(tmp_path, {
        'operation': 'point_plane_distance', 'task_id': 't1',
        'point_id': point['result_id'], 'plane_id': fit_mixed['plane_id']})
    assert query['status'] == 'ok' and query['calibration_match'] is True
    assert {(record['role'], record['camera_serial'], record['calibration_id'])
            for record in query['calibration_provenance']} == {
        ('point_id', 'cam-1', 'calib-1'),
        ('plane_id', 'cam-1', 'calib-1'),
        ('plane_id', 'cam-2', 'calib-2')}


def test_plane_result_id_tracks_calibration_provenance(tmp_path):
    """Reusing identical points under another calibration yields a distinct stored plane."""
    _task(tmp_path)
    first = _observation(tmp_path, 'cam-a', _flat(), serial='cam-1', calibration_id='calib-1')
    fit = scene_geometry.compute(tmp_path, {
        'operation': 'fit_plane', 'task_id': 't1', 'observation_path': first,
        'roi': [10, 10, 40, 30], 'stride': 2, 'voxel_size_m': 0.02,
        'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
        'calibration_id': 'calib-1'})
    assert fit['status'] == 'ok'
    recalibrated = _observation(tmp_path, 'cam-a2', _flat(), serial='cam-1',
                                calibration_id='calib-1b')
    updated = scene_geometry.compute(tmp_path, {
        'operation': 'fit_plane', 'task_id': 't1', 'observation_path': recalibrated,
        'roi': [10, 10, 40, 30], 'stride': 2, 'voxel_size_m': 0.02,
        'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
        'calibration_id': 'calib-1b'})
    assert updated['status'] == 'ok'
    assert updated['plane_id'] != fit['plane_id']
    assert updated['calibration_provenance'][0]['calibration_id'] == 'calib-1b'
    stored = json.loads((tmp_path / fit['result_path']).read_text(encoding='utf-8'))
    assert stored['calibration_provenance'][0]['calibration_id'] == 'calib-1'


# --------------------------------------------------------------------------- #
# fit_plane / distances
# --------------------------------------------------------------------------- #
def _fit_request(observation, **overrides):
    request = {'operation': 'fit_plane', 'task_id': 't1', 'observation_path': observation,
               'roi': [10, 10, 40, 30], 'stride': 2, 'voxel_size_m': 0.02,
               'base_from_camera': _transform(translation=(0.1, 0.2, 0.3)),
               'calibration_id': 'calib-1'}
    request.update(overrides)
    return request


def test_fit_plane_known_plane_and_signed_distance_conventions(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    fit = scene_geometry.compute(tmp_path, _fit_request(observation))
    assert fit['status'] == 'ok'
    assert fit['normal'] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert fit['point_on_plane_m'][2] == pytest.approx(1.3)
    assert fit['orientation']['convention'] == 'canonical_largest_component_positive'
    assert fit['residual_m']['rms'] < 1e-9
    assert fit['support']['inlier_fraction'] == 1.0
    assert 'not a total accuracy' in fit['accuracy_note']
    assert fit['result_id'] == fit['plane_id']

    raised = _observation(tmp_path, 'o2', _flat(1020), calibration_id='calib-1')
    point = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=raised, pixel=[32, 24],
        base_from_camera=_transform(translation=(0.1, 0.2, 0.3)), calibration_id='calib-1'))
    query = scene_geometry.compute(tmp_path, {
        'operation': 'point_plane_distance', 'task_id': 't1',
        'point_id': point['result_id'], 'plane_id': fit['plane_id']})
    assert query['status'] == 'ok'
    assert query['signed_distance_m'] == pytest.approx(0.02, abs=1e-9)
    assert query['absolute_distance_m'] == pytest.approx(0.02, abs=1e-9)
    assert query['side'] == 'normal_side'
    assert query['frame'] == 'base' and query['units'] == 'm'
    assert query['clearance_verified'] is False
    assert 'dot(normal, p - point_on_plane_m)' in query['signed_distance_convention']

    flipped = scene_geometry.compute(tmp_path, {
        'operation': 'point_plane_distance', 'task_id': 't1',
        'point_id': point['result_id'], 'plane_id': fit['plane_id'],
        'reference_normal': [0.0, 0.0, -1.0]})
    assert flipped['signed_distance_m'] == pytest.approx(-0.02, abs=1e-9)
    assert flipped['side'] == 'opposite_side'
    assert flipped['orientation']['normal_flipped_for_query'] is True

    down = scene_geometry.compute(tmp_path, _fit_request(observation,
                                                         reference_normal=[0.0, 0.0, -1.0]))
    assert down['normal'] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
    assert down['orientation']['convention'] == 'aligned_to_reference_normal'
    down_query = scene_geometry.compute(tmp_path, {
        'operation': 'point_plane_distance', 'task_id': 't1',
        'point_id': point['result_id'], 'plane_id': down['plane_id']})
    assert down_query['signed_distance_m'] == pytest.approx(-0.02, abs=1e-9)
    assert down_query['orientation']['convention'] == 'aligned_to_reference_normal'


def test_fit_plane_recovers_tilted_plane(tmp_path):
    _task(tmp_path)
    columns = np.arange(WIDTH)[None, :]
    plane = 1.0 / (1.0 - 0.2 * (columns - PPX) / 50.0)  # camera-frame plane z = 1 + 0.2 x
    depth = np.round(np.broadcast_to(plane, (HEIGHT, WIDTH)) / 0.001).astype(np.uint16)
    observation = _observation(tmp_path, 'tilt', depth, calibration_id='calib-1')
    fit = scene_geometry.compute(tmp_path, _fit_request(
        observation, roi=[8, 8, 56, 40], stride=2, voxel_size_m=0.01,
        base_from_camera=_transform(), calibration_id='calib-1'))
    assert fit['status'] == 'ok'
    expected = np.array([0.2, 0.0, -1.0])
    expected = expected / np.linalg.norm(expected)
    assert abs(abs(float(np.dot(fit['normal'], expected))) - 1.0) < 1e-3
    assert fit['residual_m']['rms'] < 0.002
    assert fit['support']['inlier_fraction'] > 0.9
    assert fit['quality']['thickness_rms_m'] < 0.002


def test_fit_plane_rejects_degenerate_support(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'line', _flat(), calibration_id='calib-1')
    result = scene_geometry.compute(tmp_path, _fit_request(
        observation, roi=[32, 10, 33, 40], stride=1, voxel_size_m=0.002))
    assert result['status'] == 'unavailable'
    assert 'degenerate' in result['reason'] or 'collinear' in result['reason']


def test_point_distance_uses_stored_ids_only(tmp_path):
    _task(tmp_path)
    transform = _transform(translation=(0.1, 0.2, 0.3))
    first = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    second = _observation(tmp_path, 'o2', _flat(1200), calibration_id='calib-1')
    near = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=first, pixel=[32, 24],
        base_from_camera=transform, calibration_id='calib-1'))
    far = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=second, pixel=[32, 24],
        base_from_camera=transform, calibration_id='calib-1'))
    result = scene_geometry.compute(tmp_path, {
        'operation': 'point_distance', 'task_id': 't1',
        'point_id': near['result_id'], 'other_point_id': far['result_id']})
    assert result['status'] == 'ok'
    assert result['distance_m'] == pytest.approx(0.2)
    assert result['delta_m'] == pytest.approx([0.0, 0.0, 0.2])
    assert result['clearance_verified'] is False
    assert 'not a clearance' in result['notes'][0]

    with pytest.raises(ValueError, match='raw coordinates'):
        scene_geometry.compute(tmp_path, {
            'operation': 'point_distance', 'task_id': 't1', 'point_id': near['result_id'],
            'other_point_id': far['result_id'], 'xyz_m': [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match='stored point result'):
        scene_geometry.compute(tmp_path, {
            'operation': 'point_distance', 'task_id': 't1',
            'point_id': scene_geometry.compute(tmp_path, _fit_request(first))['plane_id'],
            'other_point_id': near['result_id']})
    uncalibrated = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=first, pixel=[32, 24]))
    with pytest.raises(ValueError, match='frames differ'):
        scene_geometry.compute(tmp_path, {
            'operation': 'point_distance', 'task_id': 't1',
            'point_id': uncalibrated['result_id'], 'other_point_id': near['result_id']})
    other_camera = _observation(tmp_path, 'o3', _flat(), serial='cam-2')
    foreign = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=other_camera, pixel=[32, 24]))
    with pytest.raises(ValueError, match='same observation'):
        scene_geometry.compute(tmp_path, {
            'operation': 'point_distance', 'task_id': 't1',
            'point_id': uncalibrated['result_id'], 'other_point_id': foreign['result_id']})


def test_base_frame_cross_camera_distance_keeps_own_calibrations(tmp_path):
    """Base-frame results from different cameras legitimately use different calibrations."""
    _task(tmp_path)
    first = _observation(tmp_path, 'cam-a', _flat(), serial='cam-1', calibration_id='calib-1')
    second = _observation(tmp_path, 'cam-b', _flat(), serial='cam-2', calibration_id='calib-2')
    near = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=first, pixel=[32, 24],
        base_from_camera=_transform(), calibration_id='calib-1'))
    far = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=second, pixel=[32, 24],
        base_from_camera=_transform(translation=(0.0, 0.0, 0.25)), calibration_id='calib-2'))
    result = scene_geometry.compute(tmp_path, {
        'operation': 'point_distance', 'task_id': 't1',
        'point_id': near['result_id'], 'other_point_id': far['result_id']})
    assert result['status'] == 'ok'
    assert result['distance_m'] == pytest.approx(0.25)
    assert result['calibration_match'] is False
    assert 'different calibrations' in result['notes'][1]
    mapping = {(record['role'], record['camera_serial'], record['calibration_id'])
               for record in result['calibration_provenance']}
    assert mapping == {('point_id', 'cam-1', 'calib-1'),
                       ('other_point_id', 'cam-2', 'calib-2')}

    # the stored point keeps the mapping a trusted parent needs to revalidate
    stored = json.loads((tmp_path / near['result_path']).read_text(encoding='utf-8'))
    assert stored['calibration_provenance'] == [{
        'camera_serial': 'cam-1', 'calibration_id': 'calib-1',
        'observation': near['source']['observation'],
        'depth_digest': near['source']['depth_digest']}]

    # the same rule holds for point/plane queries across cameras
    plane = scene_geometry.compute(tmp_path, _fit_request(first))
    assert plane['status'] == 'ok' and plane['calibration_id'] == 'calib-1'
    query = scene_geometry.compute(tmp_path, {
        'operation': 'point_plane_distance', 'task_id': 't1',
        'point_id': far['result_id'], 'plane_id': plane['plane_id']})
    assert query['status'] == 'ok' and query['calibration_match'] is False
    assert {(record['role'], record['camera_serial'], record['calibration_id'])
            for record in query['calibration_provenance']} == {
        ('point_id', 'cam-2', 'calib-2'), ('plane_id', 'cam-1', 'calib-1')}

    # the same declared serial under a new calibration is a changed registration
    moved = _observation(tmp_path, 'cam-a2', _flat(), serial='cam-1', calibration_id='calib-1b')
    recalibrated = scene_geometry.compute(tmp_path, _request(
        'surface_point', observation_path=moved, pixel=[32, 24],
        base_from_camera=_transform(), calibration_id='calib-1b'))
    assert recalibrated['result_id'] != near['result_id']
    with pytest.raises(ValueError, match='conflicting calibrations for camera serial cam-1'):
        scene_geometry.compute(tmp_path, {
            'operation': 'point_distance', 'task_id': 't1',
            'point_id': near['result_id'], 'other_point_id': recalibrated['result_id']})


def test_accumulate_region_point_cap_is_enforced(tmp_path):
    _task(tmp_path)
    depth = np.full((400, 400), 1000, dtype=np.uint16)
    observation = _observation(tmp_path, 'big', depth, ppx=200.0, ppy=200.0,
                               calibration_id='calib-1')
    result = scene_geometry.compute(tmp_path, {
        'operation': 'accumulate_region', 'task_id': 't1', 'target_id': 'table',
        'batch_id': 'huge', 'voxel_size_m': 0.0001, 'stride': 1,
        'observations': [{'observation_path': observation, 'roi': [0, 0, 400, 400],
                          'base_from_camera': _transform(), 'calibration_id': 'calib-1'}]})
    assert result['status'] == 'unavailable'
    assert '100000-point cap' in result['reason']
    assert not (tmp_path / 'tasks' / 't1' / 'geometry' / 'targets' / 'table' / 'huge.npz').exists()


def test_fit_plane_degeneracy_and_support_limits(tmp_path):
    _task(tmp_path)
    observation = _observation(tmp_path, 'o1', _flat(), calibration_id='calib-1')
    result = scene_geometry.compute(tmp_path, _fit_request(
        observation, ransac={'min_inliers': 5000}))
    assert result['status'] == 'unavailable'
    assert 'inadequate support' in result['reason']
    result = scene_geometry.compute(tmp_path, _fit_request(
        observation, reference_normal=[1.0, 0.0, 0.0]))
    assert result['status'] == 'unavailable'
    assert 'ambiguous' in result['reason']


def test_geometry_output_cannot_escape_via_task_symlink(tmp_path):
    _task(tmp_path)
    obs = _observation(tmp_path, 'o1', _flat())
    outside=tmp_path/'outside';outside.mkdir()
    geometry=tmp_path/'tasks/t1/geometry';geometry.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        scene_geometry.compute(tmp_path,_request('surface_point',observation_path=obs,pixel=[32,24]))
    assert not list(outside.iterdir())


def test_region_summary_filters_background_using_stored_support_plane(tmp_path):
    _task(tmp_path)
    depth=_flat();depth[12:18,14:24]=1100
    observation=_observation(tmp_path,'object',depth,calibration_id='calib-1')
    transform=_transform(translation=(.1,.2,.3))
    plane=scene_geometry.compute(tmp_path,_request('fit_plane',observation_path=observation,
        roi=[32,8,62,40],base_from_camera=transform,calibration_id='calib-1'))
    assert plane['status']=='ok'
    entry={'observation_path':observation,'roi':[10,10,30,20],
           'base_from_camera':transform,'calibration_id':'calib-1'}
    result=scene_geometry.compute(tmp_path,_request('accumulate_region',target_id='object',batch_id='foreground',
        plane_id=plane['plane_id'],observations=[entry],voxel_size_m=.002))
    assert result['status']=='ok' and result['support_plane_id']==plane['plane_id']
    assert result['surface_summary']['median_point_m'][2]==pytest.approx(1.4)
    points=np.load(tmp_path/result['points_npz'])
    assert len(points['points_base_m'])==len(points['entry_index'])
    assert np.allclose(points['points_base_m'][:,2],1.4)
