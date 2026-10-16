"""Local compute migration must keep exactly the original helper semantics."""
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_longvil_physics import helper_argv
from run_argon_video_study import transport


def args(host):
    return SimpleNamespace(host=host,port=2222,socket=Path('/tmp/control'),
        remote_root='/tmp/source tree',remote_python='/python',remote_helper='/helper.py')


def test_local_helper_does_not_use_ssh_or_shell():
    command=helper_argv(args('local'),'start','/profile space','/scene.xml')
    assert command[:3]==['env','MUJOCO_GL=egl','PYOPENGL_PLATFORM=egl']
    assert command[-6:]==['/helper.py','start','--root','/profile space','--scene','/scene.xml']
    assert 'ssh' not in command and 'bash' not in command


def test_ssh_wraps_identical_local_argv():
    import shlex
    command=helper_argv(args('host'),'score','/profile space')
    assert command[0]=='ssh'
    assert shlex.split(command[-1])==helper_argv(args('local'),'score','/profile space')


def test_local_scene_validation_uses_local_command_runner():
    assert transport({'host':'local'})==['bash','-lc']
