"""Public experiment memory: lossless disk archive, small deterministic working set.

No provider summaries, hidden reasoning, private scoring, or robot execution.
Historical image processing never advances the live observation identity.
"""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import uuid


def write_json(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2).encode()
    temp = path.with_suffix('.tmp')
    temp.write_bytes(data)
    temp.replace(path)
    return hashlib.sha256(data).hexdigest()


CONTEXT_TOOL = {
    'name': 'session_context',
    'description': 'READ ONLY public session memory. list paginates archived turns; read returns a turn and exact historical image paths; crop processes an archived image ROI and returns pixels with original coordinates. Turn numbers are zero-based. This never moves the robot, recaptures the scene or makes old evidence current.',
    'inputSchema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['operation'],
        'properties': {
            'operation': {'enum': ['list', 'read', 'crop']},
            'turn': {'type': 'integer', 'minimum': 0},
            'start': {'type': 'integer', 'minimum': 0, 'default': 0},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 10},
            'image_id': {'type': 'string', 'minLength': 1},
            'roi': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 4, 'maxItems': 4},
        },
        'allOf': [
            {'if': {'properties': {'operation': {'const': 'read'}}}, 'then': {'required': ['turn']}},
            {'if': {'properties': {'operation': {'const': 'crop'}}}, 'then': {'required': ['image_id', 'roi']}},
        ],
    },
}


class SessionContext:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / 'context'
        self.root.mkdir(exist_ok=False)
        (self.root / 'turns').mkdir()
        self.session_id = str(uuid.uuid4())
        self.images = {}
        self.turns = {}
        self.files = {}  # Explicit public text/image allowlist; never scan the workspace.
        self.contract_path = self.root / 'tools.json'
        self.current_path = self.root / 'current.json'
        self.index_path = self.root / 'index.json'

    def publish(self, path, value):
        self.files[str(path)] = write_json(path, value)

    def register_images(self, turn, manifest):
        for item in manifest:
            path = Path(item['path'])
            if path.parent != self.workspace or path.is_symlink():
                raise ValueError('Image outside public session workspace')
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Public image hash mismatch')
            ident = path.name
            if ident not in self.images:
                self.images[ident] = {**copy.deepcopy(item), 'image_id': ident, 'first_turn': turn,
                                      'scope': 'Archived capture, not proof of current scene state'}
            elif self.images[ident]['sha256'] != item['sha256']:
                raise ValueError('Immutable image identity changed')
            self.files[str(path)] = item['sha256']

    def archive(self, turn, messages, manifest, tools, capsule):
        if turn in self.turns:
            raise ValueError('Turn already archived; do not overwrite a prior decision')
        self.register_images(turn, manifest)
        path = self.root / 'turns' / f'turn-{turn:03d}.json'
        # Only model-visible content supplied by the runner, not private action-chain snapshots.
        record = {'session_id': self.session_id, 'turn': turn, 'scope': 'HISTORICAL public evidence; may be stale',
                  'messages': copy.deepcopy(messages), 'images': copy.deepcopy(manifest)}
        self.publish(path, record)
        self.turns[turn] = {'turn': turn, 'path': str(path), 'images': [m['path'].split('/')[-1] for m in manifest]}
        self.publish(self.contract_path, {'tools': tools, 'scope': 'Authoritative input schemas; do not infer names from memory'})
        self.publish(self.current_path, capsule)
        self.publish(self.index_path, {'session_id': self.session_id, 'turn_numbering': 'zero-based',
                                      'turns': list(self.turns.values()), 'images': list(self.images.values())})

    def record_decision(self, turn, decision):
        path = self.root / 'turns' / f'turn-{turn:03d}-decision.json'
        if path.exists():
            raise ValueError('Decision already archived')
        self.publish(path, {'turn': turn, 'model_candidate': decision,
                            'scope': 'Model proposal only; execution feedback is in a subsequent turn'})
        self.turns[turn]['decision_path'] = str(path)
        self.publish(self.index_path, {'session_id': self.session_id, 'turn_numbering': 'zero-based',
                                      'turns': list(self.turns.values()), 'images': list(self.images.values())})

    def record_tool_result(self, turn, value):
        """Persist actual feedback, including the last tool before session exit."""
        row = self.turns[turn]
        path = self.root / 'turns' / f'turn-{turn:03d}-results.json'
        values = self.read_public(path) if str(path) in self.files else []
        values.append(copy.deepcopy(value))
        self.publish(path, values)
        row['results_path'] = str(path)
        self.publish(self.index_path, {'session_id':self.session_id, 'turn_numbering':'zero-based',
                                      'turns':list(self.turns.values()), 'images':list(self.images.values())})

    def read_public(self, path):
        path = Path(path)
        expected = self.files.get(str(path))
        if not expected or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Cached public evidence changed or unavailable')
        return json.loads(path.read_text())

    def query(self, args):
        """Return (JSON receipt, optional multimodal parts). Never touches live gate."""
        from jsonschema import validate
        validate(args, CONTEXT_TOOL['inputSchema'])
        op = args['operation']
        if op == 'list':
            start, limit = args.get('start', 0), args.get('limit', 10)
            rows = list(self.turns.values())
            return {'session_id': self.session_id, 'turns': rows[start:start+limit],
                    'total': len(rows), 'next_start': start+limit if start+limit < len(rows) else None,
                    'scope': 'HISTORICAL session index'}, []
        if op == 'read':
            row = self.turns.get(args['turn'])
            if row is None:
                raise ValueError('Unknown cached turn; use session_context list')
            result = self.read_public(row['path'])
            if 'decision_path' in row:
                result['decision'] = self.read_public(row['decision_path'])
            if 'results_path' in row:
                result['actual_tool_results'] = self.read_public(row['results_path'])
            return result, []
        item = self.images.get(args['image_id'])
        if item is None:
            raise ValueError('Unknown image_id; read the session index or turn first')
        path = Path(item['path'])
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Cached image changed; processing refused')
        from PIL import Image
        with Image.open(path) as im:
            x0, y0, x1, y1 = args['roi']
            if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
                raise ValueError('ROI must lie inside the original image: [x0,y0,x1,y1]')
            image = im.crop((x0, y0, x1, y1)).convert('RGB')
            output = io.BytesIO(); image.save(output, format='PNG')
            original_size = list(im.size)
        result = {'status': 'ok', 'scope': 'HISTORICAL derived image; NOT a new live observation',
                  'source_image_id': item['image_id'], 'source_sha256': item['sha256'],
                  'source_label': item['label'], 'source_turn': item['first_turn'],
                  'roi': args['roi'], 'original_size': original_size,
                  'pixel_mapping': 'original_x = crop_x + roi[0]; original_y = crop_y + roi[1]',
                  'calibration_scope': 'No new metric geometry or live calibration inferred'}
        return result, [{'type': 'text', 'text': 'HISTORICAL CACHE CROP: '+json.dumps(result)},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()}}]
