"""The relay must exit cleanly even if its client's stdin remains open."""
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.skipif(sys.platform=='win32',reason='Host-side Parallels relay test uses a POSIX executable')
def test_blank_trailer_and_open_stdin_do_not_crash_shutdown(tmp_path):
    fake=tmp_path/'prlctl'
    fake.write_text('#!/usr/bin/env python3\nprint(\'{"jsonrpc":"2.0","id":1,"result":{}}\')\nprint()\n')
    fake.chmod(0o755)
    relay=Path(__file__).resolve().parents[2]/'tools/parallels_mcp.py'
    p=subprocess.Popen([sys.executable,str(relay),'--prlctl',str(fake),'--python','unused','--root','unused'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        assert p.wait(timeout=5)==0
        assert p.stdout.read()==b'{"jsonrpc":"2.0","id":1,"result":{}}\n'
        assert b'_enter_buffered_busy' not in p.stderr.read()
    finally:
        p.stdin.close()
        if p.poll() is None:p.terminate();p.wait(timeout=5)


@pytest.mark.skipif(sys.platform=='win32',reason='Host-side signal test')
def test_sigterm_closes_owned_child_input(tmp_path):
    fake=tmp_path/'prlctl';closed=tmp_path/'closed'
    fake.write_text('#!/usr/bin/env python3\nimport sys,pathlib\nprint("ready",flush=True)\nsys.stdin.buffer.read()\npathlib.Path('+repr(str(closed))+').write_text("EOF")\n')
    fake.chmod(0o755)
    relay=Path(__file__).resolve().parents[2]/'tools/parallels_mcp.py'
    p=subprocess.Popen([sys.executable,str(relay),'--prlctl',str(fake),'--python','unused','--root','unused'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        assert p.stdout.readline()==b'ready\n'
        p.terminate();assert p.wait(timeout=8)==143
        assert closed.read_text()=='EOF'
    finally:
        p.stdin.close()
        if p.poll() is None:p.kill();p.wait(timeout=5)
