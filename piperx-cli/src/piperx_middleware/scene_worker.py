"""Single CPU query process; contains no robot transports."""
import json
from pathlib import Path
import sys
from .scene_geometry import compute


def main():
    raw = sys.stdin.buffer.read(65537)
    if len(raw)>65536:
        raise ValueError('Geometry request exceeds 64 KiB')
    from .process_lock import ProcessLock
    root = Path(sys.argv[1]).resolve()
    lock = ProcessLock(root/'geometry.lock')
    try:
        result = compute(root,json.loads(raw))
    finally:
        lock.close()
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
