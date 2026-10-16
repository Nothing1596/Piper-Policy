"""Experiment-only video lessons. No robot connection, GT, or test-result input.

Two independent CLI sessions propose and then revise lessons using different
decoded frames. The output is observational evidence, never a physics certificate.
"""
import asyncio
import hashlib
import json
from pathlib import Path

from agy_online_decider import AgyDecider, usage_delta, save, interrupted_generation


LESSON_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['lessons'],
    'properties': {'lessons': {'type': 'array', 'maxItems': 3, 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['when', 'guidance', 'verification', 'uncertainty', 'evidence_refs'],
        'properties': {**{k: {'type': 'string', 'minLength': 1, 'maxLength': 500}
                          for k in ('when', 'guidance', 'verification', 'uncertainty')},
                       'evidence_refs': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                                         'items': {'type': 'string', 'minLength': 1}}}}}}
}


def validate_lessons(value, allowed_refs):
    from jsonschema import validate
    validate(value, LESSON_SCHEMA)
    for lesson in value['lessons']:
        if not set(lesson['evidence_refs']) <= set(allowed_refs):
            raise ValueError('Lesson cites undelivered video evidence')
    return value


def ensure_disjoint(training, testing):
    """Reject both task identity overlap and re-encoded entries with known same hash."""
    if not training or not testing:
        raise ValueError('Both learning and evaluation splits are required')
    for key in ('task_id', 'source_sha256'):
        if any(not row.get(key) for row in training + testing):
            raise ValueError('Missing split identity: ' + key)
        if {r[key] for r in training} & {r[key] for r in testing}:
            raise ValueError('Training/test overlap: ' + key)


async def video_pass(root, frames, *, model, previous=None):
    """Reuse only the audited CLI transport and exact-image allowlist."""
    if not frames or any(not f.get('evidence_id') for f in frames):
        raise ValueError('Nonempty, identified evidence required')
    if len({f['evidence_id'] for f in frames}) != len(frames):
        raise ValueError('Duplicate evidence identities')
    root = Path(root)
    d = AgyDecider(root, model=model, exploration=True)
    save(root/'response-schema.json', LESSON_SCHEMA)
    manifest = []
    allow = {}
    for i, frame in enumerate(frames):
        data = Path(frame['image_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != frame['image_sha256']:
            raise ValueError('Changed training frame')
        target = d.workspace/'images'/f'frame-{i:03d}.jpg'
        target.write_bytes(data)
        allow[str(target)] = hashlib.sha256(data).hexdigest()
        manifest.append({'id': frame['evidence_id'], 'time_s': frame['timestamp_s'],
                         'path': str(target), 'source_type': 'recorded_training_video'})
    save(d.state_path, {'turn': 0, 'images': allow, 'audit': str(d.audit_path)})
    prompt = (
        'You learn reusable observational guidance from recorded human videos. '
        'Use native view_file to actually inspect EVERY exact image path below. '
        'Only these image reads and structured finish are allowed. No shell, other files, '
        'network, delegation, robot or simulator. No private reasoning is requested. '
        'Produce at most THREE short lessons with applicability, guidance, verification, '
        'uncertainty and exact evidence_refs. Empty lessons are valid if evidence is weak. '
        'Avoid object colors, instance names, task-specific sequences or metric poses as rules. '
        'Distinguish observed order from causal necessity. Video cannot establish force, '
        'friction, robot reachability or guaranteed success. Preserve those uncertainties. '
        'Never treat a proposed lesson as a proven physical skill. '
    )
    if previous is None:
        prompt += 'Propose provisional lessons from the initial overview. '
    else:
        prompt += ('Check the prior hypotheses against the supplemental decoded frames. '
                   'Revise, narrow, or remove unsupported claims. Cite supplemental evidence '
                   'where it actually changes or corroborates a claim. This is evidence-based '
                   'revision, not independent ground truth. Prior hypotheses:\n' +
                   json.dumps(previous, ensure_ascii=False))
    prompt += '\nDelivered recorded evidence:\n' + json.dumps(manifest, ensure_ascii=False)
    (root/'prompt.txt').write_text(prompt)
    try:
        failed_usage=[];failed_invocations=0
        for attempt in range(4):
            result = await asyncio.wait_for(d.request(prompt), 600)
            save(root/f'attempt-{attempt:02d}-result.json', result)
            save(root/'result.json', result)
            usage = usage_delta(result.get('usage'), {})
            if not interrupted_generation(result):
                break
            rows=[json.loads(x) for x in d.audit_path.read_text().splitlines()]
            failed_invocations += sum(r['phase']=='invocation' and
                                      r['conversation_id']==d.conversation for r in rows)
            failed_usage.append(usage)
            save(root/f'attempt-{attempt:02d}-interruption.json',
                 {'usage':usage,'conversation_id':d.conversation,'action_dispatched':False,
                  'error':result.get('error'),'retry':attempt<3})
            if attempt==3:
                raise ValueError('Video generation interruption retry limit reached')
            await d.close()
            d.turn+=1
            save(d.state_path, {'turn':d.turn,'images':allow,'audit':str(d.audit_path)})
            await asyncio.sleep(5)
        usage={k:usage[k]+sum(u[k] for u in failed_usage) for k in usage}
        if not d.conversation or result.get('conversation_id') != d.conversation:
            raise ValueError('Video session identity mismatch')
        rows = [json.loads(x) for x in d.audit_path.read_text().splitlines()]
        rows = [r for r in rows if r['conversation_id'] == d.conversation]
        seen = {r['path'] for r in rows if r['phase'] == 'post' and r['allowed']
                and r['tool'] == 'view_file' and not r.get('error')}
        receipt = {'usage': usage, 'requested_model': model,
                   'runtime_models': sorted({r['model'] for r in rows if r.get('model')}),
                   'required_images': list(allow), 'confirmed_images': sorted(seen),
                   'model_invocations': failed_invocations+sum(r['phase'] == 'invocation' for r in rows),
                   'generation_interruptions':len(failed_usage),
                   'blocked_tool_calls': sum(r['phase'] == 'pre' and not r['allowed'] for r in rows),
                   'conversation_id': result.get('conversation_id')}
        save(root/'receipt.json', receipt)
        if result.get('status') != 'SUCCESS' or set(allow) - seen:
            raise ValueError('Video pass unsuccessful or image delivery unconfirmed')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in allow.items()):
            raise ValueError('Evidence changed during viewing')
        value = result.get('structured_output')
        if value is None:
            value = json.loads(result['response'])
        return validate_lessons(value, [f['evidence_id'] for f in frames])
    finally:
        await d.close()


async def learn(root, training, testing, model):
    ensure_disjoint(training, testing)
    if len(training) != 3:
        raise ValueError('This pilot freezes exactly three learning videos')
    root = Path(root)
    root.mkdir(exist_ok=False, parents=True)
    lessons = []
    for i, video in enumerate(training):
        initial_ids = {f['evidence_id'] for f in video['initial']}
        supplemental = {f['evidence_id'] for f in video['supplement']} - initial_ids
        if not supplemental:
            raise ValueError('Revision requires additional evidence')
        first = await video_pass(root/f'video-{i:02d}-initial', video['initial'], model=model)
        # The reviser sees both packets so original claims remain auditable.
        combined = {f['evidence_id']: f for f in video['initial'] + video['supplement']}
        revised = await video_pass(root/f'video-{i:02d}-revision', list(combined.values()),
                                   model=model, previous=first)
        for lesson in revised['lessons']:
            lessons.append({**lesson, 'id': f'v{i:02d}-l{len(lessons):02d}',
                            'source_sha256': video['source_sha256'],
                            'status': 'video_supported_not_execution_validated'})
    receipts = [json.loads(p.read_text()) for p in sorted(root.glob('*/receipt.json'))]
    if len({r['conversation_id'] for r in receipts}) != 6:
        raise ValueError('Learning passes must have distinct provider sessions')
    library = {'version': 1, 'model': model, 'learning_simulator_actions': 0,
               'learning_hardware_actions': 0, 'lessons': lessons,
               'training_sources': [v['source_sha256'] for v in training],
               'test_sources': [v['source_sha256'] for v in testing],
               'usage': {k: sum(r['usage'][k] for r in receipts) for k in receipts[0]['usage']},
               'evidence_checks': 'All frames viewed; citations valid. Semantic support remains a model judgment.'}
    save(root/'library.json', library)
    (root/'library.sha256').write_text(hashlib.sha256((root/'library.json').read_bytes()).hexdigest())
    return library


def render_library(path, expected_hash):
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError('Frozen video experience changed')
    lib = json.loads(data)
    # Stable, bounded experiment packet; no retrieval model or test feedback.
    if len(lib['lessons']) > 9:
        raise ValueError('Pilot experience packet exceeds nine lessons')
    return ('VIDEO-DERIVED EXPERIENCE (frozen before evaluation; observational, not '
            'physically validated): Apply only when current evidence supports the conditions. '
            'Current observations override historical expectations. These are soft guidance, '
            'not compulsory actions. Do not infer contact/force/coordinates from these lessons.\n'
            + json.dumps(lib['lessons'], ensure_ascii=False))
