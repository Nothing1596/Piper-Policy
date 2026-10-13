"""Pass stdio MCP through Parallels, dropping its empty stdout trailer only.

Does not parse tools, start CAN, retry actions, or suppress nonempty failures.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vm',default='Windows 11');p.add_argument('--python',required=True)
    p.add_argument('--root',required=True);p.add_argument('--mode',choices=['real','simulation'],default='real')
    p.add_argument('--prlctl',default='/usr/local/bin/prlctl')
    a=p.parse_args()
    # MCP clients terminate stdio servers with SIGTERM; run the same cleanup
    # as EOF instead of leaving the transport child behind.
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    child=subprocess.Popen([a.prlctl,'exec',a.vm,'--current-user',a.python,'-u','-m',
        'piperx_middleware.standalone_cli','--root',a.root,'--mode',a.mode,'--agent-mcp'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,bufsize=0)
    def input_reader():
        try:
            while True:
                # A daemon thread must not hold Python's buffered stdin lock
                # when the remote server exits before the client sends EOF.
                data=os.read(sys.stdin.fileno(),65536)
                if not data:break
                child.stdin.write(data);child.stdin.flush()
        except (BrokenPipeError,OSError):pass
        finally:
            try:child.stdin.close()
            except OSError:pass
    threading.Thread(target=input_reader,daemon=True).start()
    try:
        for line in child.stdout:
            if line.strip():sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
        return child.wait()
    finally:
        if child.poll() is None:
            child.stdin.close()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)


if __name__=='__main__':raise SystemExit(main())
