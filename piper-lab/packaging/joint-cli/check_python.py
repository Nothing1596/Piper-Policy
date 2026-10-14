import struct
import sys

if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
    raise SystemExit('Python 3.12 x64 required')
