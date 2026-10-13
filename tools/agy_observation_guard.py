"""Experiment-only agy hook: exact public cache reads, no shell/code/private data."""
import hashlib
import json
from pathlib import Path
import sys
import time


def permitted(call, allow):
    if call.get('name') == 'finish':
        # agy's internal structured-output tool returns data, never executes it.
        # The coordinator validates its JSON schema before dispatching one request.
        return True, None
    if call.get('name') != 'view_file':
        return False, None
    raw = call.get('args', {}).get('AbsolutePath')
    if not isinstance(raw, str):
        return False, None
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve()) != str(path):
        return False, None
    expected = allow.get(str(path))
    if not expected or path.suffix not in ('.jpg', '.png', '.json') or not path.is_file():
        return False, str(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)


def main():
    phase, state_path = sys.argv[1:]
    state_path = Path(state_path)
    data = json.load(sys.stdin)
    state = json.loads(state_path.read_text())
    ok, path = permitted(data.get('toolCall', {}), state['images'])
    row = dict(phase=phase, turn=state['turn'], at=time.time(),
               conversation_id=data.get('conversationId'), model=data.get('modelName'),
               step=data.get('stepIdx'), tool=data.get('toolCall', {}).get('name'),
               path=path, allowed=ok, error=data.get('error'))
    with Path(state['audit']).open('a') as out:
        out.write(json.dumps(row)+'\n')
    if phase == 'pre':
        allowed_paths = list(state['images'])
        suggested = [p for p in allowed_paths if path and Path(p).name==Path(path).name]
        print(json.dumps({'decision': 'allow' if ok else 'deny',
                          'reason': 'Only hash-approved public cache/image reads and the structured-output finish tool are allowed. No shell, private GT, or other sessions. Recover contracts/paths from: '+json.dumps(state.get('recovery',{}))
                          + (' Copy this exact approved image path: '+json.dumps(suggested) if suggested and not ok else '')}))
    else:
        print('{}')


if __name__ == '__main__':
    main()
