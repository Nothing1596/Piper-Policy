"""Native MuJoCo geometry checks; no render window, API key or robot hardware."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
pytest.importorskip('mujoco')
from longvil_physics_host import scene_backend_class,process_identity,stop_owned
from longvil_physics_scene import build_scene,complete_scene_metadata


def test_process_identity_and_pid_reuse_never_signal_a_foreign_process(tmp_path,monkeypatch):
    import json,longvil_physics_host as host
    proc=tmp_path/'proc/10';proc.mkdir(parents=True)
    fields=['S']+['0']*18+['12345']+['0']*5
    (proc/'stat').write_text('10 (name with spaces) '+' '.join(fields))
    (proc/'cmdline').write_bytes(b'python\0helper\0serve\0--root\0profile\0')
    identity=process_identity(10,tmp_path/'proc');assert identity['start_ticks']=='12345'
    profile=tmp_path/'profile';profile.mkdir();(profile/'physics-owner.json').write_text(json.dumps(identity))
    calls=[];monkeypatch.setattr(host.os,'kill',lambda *args:calls.append(args))
    monkeypatch.setattr(host,'process_identity',lambda pid:dict(identity,start_ticks='54321'))
    with pytest.raises(RuntimeError,match='identity changed'):stop_owned(profile)
    assert not calls


def test_grasp_width_follows_measured_pad_axis_and_initial_seating(tmp_path):
    assets=tmp_path/'assets/buildings';assets.mkdir(parents=True)
    (assets/'cuboid.urdf').write_text('<robot name="test"><link name="root"><collision><geometry><box size=".04 .08 .04"/></geometry></collision><inertial><mass value=".1"/></inertial></link></robot>')
    ann={'object_list':['green cuboid'],'positions':{'green cuboid':{'x':0.,'y':-.5,'z':.01}}}
    template=Path(__file__).parents[2]/'piperx-cli/assets/mujoco/scene.xml';scene=tmp_path/'scene.xml'
    build_scene(ann,template,assets.parent,scene,translation_m=[-.2,0,-.2]);meta=complete_scene_metadata(scene)
    backend=scene_backend_class()(str(scene))
    try:
        row=backend.evaluate()['objects']['green cuboid'];grasp=row['grasp_frame']
        assert np.isclose(grasp['closing_width_m'],.04) and grasp['yaw_deg']==0.
        assert np.isclose(row['aabb']['lower'][2],-.198) and row['grasp_frame']['footprint_xy_m']
        # Recorder takes renderer then state lock; nested observe is reentrant.
        with backend.render_lock,backend.lock:
            with backend.render_lock:backend.evaluate()
        assert meta['asset_sha256s']
    finally:backend.close()
