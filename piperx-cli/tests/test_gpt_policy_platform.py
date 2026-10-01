"""Cross-platform glue tests; mocks are not native Windows execution evidence."""
import errno
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import pytest
from gpt_policy_piper import platform


def test_windows_pipe_writer_unicode_with_real_pipe():
    process=subprocess.Popen([sys.executable,'-c','import sys; print(sys.stdin.readline().strip())'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    try:
        platform.write_windows_pipe(process,'你好\n',time.monotonic()+2)
        assert process.stdout.readline().strip()=='你好'
        assert process.wait(timeout=2)==0
    finally:
        if process.poll() is None:process.kill()
        for stream in (process.stdin,process.stdout,process.stderr):stream.close()


def test_windows_lock_rethrows_noncontention(tmp_path,monkeypatch):
    fake=SimpleNamespace(LK_NBLCK=1,locking=lambda *args: (_ for _ in ()).throw(OSError(errno.EBADF,'bad descriptor')))
    monkeypatch.setitem(sys.modules,'msvcrt',fake)
    monkeypatch.setattr(platform,'os',SimpleNamespace(name='nt',fstat=os.fstat,write=os.write,lseek=os.lseek,SEEK_SET=os.SEEK_SET))
    with (tmp_path/'lock').open('a+b') as lock:
        with pytest.raises(OSError) as exc:platform.acquire_cache_lock(lock.fileno())
    assert exc.value.errno==errno.EBADF


def test_windows_process_tree_command_and_timeout(monkeypatch):
    calls=[]
    process=SimpleNamespace(pid=123,poll=lambda:None,kill=lambda:calls.append('kill'),wait=lambda **kw:calls.append('wait'))
    monkeypatch.setattr(platform.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0)))
    platform.close_windows_process(process)
    assert calls[0]==['taskkill','/PID','123','/T','/F']
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('taskkill',10)
    monkeypatch.setattr(platform.subprocess,'run',timeout)
    with pytest.raises(RuntimeError,match='unconfirmed'):platform.close_windows_process(process)
    assert calls[-2:]==['kill','wait']


def test_stop_allowed_after_heartbeat_failure():
    from gpt_policy_piper.transport import SimulationTransport
    transport=SimulationTransport.__new__(SimulationTransport)
    transport._heartbeat_error=RuntimeError('gone');transport.instance_id='instance'
    calls=[]
    transport.client=SimpleNamespace(call=lambda method,path,body:(calls.append(path) or {'status':'idle'}))
    assert transport.call('POST','/v1/stop')['status']=='idle'
    with pytest.raises(RuntimeError):transport.call('POST','/v1/move')
    assert calls==['/v1/stop']


def test_windows_lock_retries_errno_only_contention(tmp_path,monkeypatch):
    attempts=[]
    def locking(*args):
        attempts.append(args)
        if len(attempts)==1:raise OSError(errno.EACCES,'lock busy')
    fake=SimpleNamespace(LK_NBLCK=1,locking=locking)
    monkeypatch.setitem(sys.modules,'msvcrt',fake)
    monkeypatch.setattr(platform,'os',SimpleNamespace(name='nt',fstat=os.fstat,write=os.write,lseek=os.lseek,SEEK_SET=os.SEEK_SET))
    with (tmp_path/'lock').open('a+b') as lock:platform.acquire_cache_lock(lock.fileno())
    assert len(attempts)==2
