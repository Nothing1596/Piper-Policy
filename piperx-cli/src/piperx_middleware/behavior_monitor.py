"""Task-scoped public behavior diagnostics. Never dispatches or gates an action."""
from collections import deque
import copy
import hashlib
import json

_READS = {'scene_query', 'grasp_target', 'placement_target', 'preview_move_to'}
_UNKNOWN = {'outcome_unknown', 'transport_unknown', 'uncertain_outcome', 'invalid_response'}
_TERMINAL = {'succeeded', 'completed', 'failed', 'cancelled', 'rejected', 'stopped', 'outcome_unknown'}


def plan_ledger(plan):
    try:
        value = json.loads((plan or {}).get('interpretation', ''))
        return value if isinstance(value, dict) and isinstance(value.get('steps'), list) else None
    except (ValueError, TypeError):
        return None


def public_receipt(result):
    result = result if isinstance(result, dict) else {}
    job = result.get('job') if isinstance(result.get('job'), dict) else result
    error = job.get('error')
    code = job.get('error_code') or (error.get('code') if isinstance(error, dict) else None)
    return {'status': job.get('status'), 'code': code,
            'request_id': job.get('request_id') or result.get('request_id'),
            'job_id': job.get('job_id') or result.get('job_id'),
            'unknown': job.get('status') == 'outcome_unknown' or code in _UNKNOWN,
            'rejected': isinstance(error, dict) or job.get('status') == 'rejected',
            'blocked': code == 'motion_blocked'}


def completion_issues(ledger, receipts):
    """References must include positive execution evidence, not only exist.

    receipts maps IDs to PUBLIC results; no object identity or GT inference.
    A successful primitive alone still does not prove a semantic transfer.
    """
    issues = []
    for step in (ledger or {}).get('steps', []):
        if not isinstance(step, dict) or step.get('status') != 'completed':
            continue
        refs = step.get('execution_refs')
        rows = [receipts.get(ref) for ref in refs if isinstance(ref, str)] if isinstance(refs, list) else []
        if not rows:
            issues.append('Completed steps need actual request/observation references; review unsupported completion claims.')
        elif any(row is None for row in rows):
            issues.append('Completed steps cite receipts unavailable in this monitor session. Read saved public outcomes to verify them; do not replay actions to recreate receipts.')
        elif not any(row.get('status') in ('succeeded', 'completed') and not row.get('unknown')
                     and not row.get('rejected') and not row.get('code') for row in rows):
            issues.append('Completed step cites only failed, blocked, rejected or uncertain outcomes; cite later successful recovery and release evidence.')
    return list(dict.fromkeys(issues))


def reviewed_prefix(plan, chain):
    """A stage-only edit cannot acknowledge actual release/blocked checkpoints."""
    ledger = plan_ledger(plan)
    if not ledger: return 0
    refs = {ref for step in ledger['steps'] if isinstance(step, dict)
            and isinstance(step.get('execution_refs'), list)
            for ref in step['execution_refs'] if isinstance(ref, str)}
    for index, row in enumerate(chain):
        result = public_receipt(row.get('actual_result'))
        body = row.get('model_arguments') or {}
        opened = row.get('tool') in ('set_gripper', 'robot_gripper') and result['status'] == 'succeeded' and body.get('width_m', 0) >= .06
        if (opened or result['blocked']) and not refs.intersection(
                {body.get('request_id'), row.get('result_observation_id')}):
            return index
    return len(chain)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _evidence(result):
    # Clock/latency/cache counters are not new decision evidence.
    return {k: result[k] for k in ('status', 'error', 'error_code', 'reason', 'point_m', 'point_base_m',
        'poses', 'surface_summary', 'normal', 'point_on_plane_m', 'distance_m', 'geometry_consistency') if k in result}


class BehaviorMonitor:
    def __init__(self):
        self.task_id = None
        self.plan = None
        self.view = None
        self.sequence = 0
        self.rows = deque(maxlen=64)
        self.receipts = {}
        self.issues = {}
        self._finished = set()
        self._jobs = {}

    def bind_task(self, task_id):
        if task_id != self.task_id:
            self.__init__()
            self.task_id = task_id

    def _raise(self, code, signature, refs, message):
        key = code + ':' + signature
        previous = self.issues.get(key)
        self.issues[key] = {**(previous or {}), 'code': code, 'signature': signature, 'state': 'needs_review',
                            'related_calls': refs[-4:], 'message': message,
                            'suggest_independent_review': bool(previous and (previous['state']=='awaiting_verification' or previous.get('suggest_independent_review'))),
                            'occurrences': (previous or {}).get('occurrences', 0) + 1}
        if len(self.issues) > 16:
            del self.issues[next(iter(self.issues))]

    def record_plan(self, plan):
        if not isinstance(plan, dict) or 'error' in plan or not plan.get('task_id'):
            return
        self.bind_task(plan['task_id'])
        if self.plan and plan.get('version', 0) < self.plan.get('version', 0):
            return
        if self.plan == plan:
            return  # Re-reading the same plan is not a second review attempt.
        old = plan_ledger(self.plan)
        new = plan_ledger(plan)
        substantive = self.plan is not None and self.plan.get('interpretation') != plan.get('interpretation')
        if substantive:
            for issue in self.issues.values():
                if issue['state'] == 'needs_review':
                    issue['state'] = 'awaiting_verification'
        if old:
            before = {s['id']: s for s in old['steps'] if isinstance(s, dict) and isinstance(s.get('id'), str)}
            after = {s['id']: s for s in (new or {}).get('steps', []) if isinstance(s, dict) and isinstance(s.get('id'), str)}
            removed = sorted(set(before) - set(after))
            # Free-form action wording is model-authored and revisable. String
            # inequality cannot establish that a demonstrated step disappeared.
            if removed:
                self._raise('plan_shrink', 'sequence', removed,
                            'Previous step IDs were removed. Cite new demo evidence and explain the revision; do not silently replace the sequence with its final state.')
                prior = self.issues['plan_shrink:sequence'].get('previous_steps', {})
                self.issues['plan_shrink:sequence']['previous_steps'] = {**{k:before[k] for k in removed}, **prior}
            elif new:
                for issue in self.issues.values():
                    if issue['code'] == 'plan_shrink' and all(k in after for k in issue.get('previous_steps',{})):
                        issue['state'] = 'resolved'
        self.plan = copy.deepcopy(plan)
        for problem in completion_issues(new, self.receipts):
            self._raise('unsupported_completion', _digest(problem), [], problem)
        if not completion_issues(new, self.receipts):
            for issue in self.issues.values():
                if issue['code'] == 'unsupported_completion': issue['state'] = 'resolved'

    def record_result(self, name, arguments, result, *, observation_id=None, result_observation_id=None):
        if not isinstance(arguments, dict) or not isinstance(result, dict): return
        self.sequence += 1
        receipt = public_receipt(result)
        request_id = receipt['request_id'] or arguments.get('request_id')
        is_poll = name == 'robot_status'
        job_id = receipt['job_id'] or (arguments.get('job_id') if is_poll else None)
        if job_id and not is_poll and arguments.get('request_id'):
            self._jobs[job_id] = (name, copy.deepcopy(arguments))
        if is_poll and job_id in self._jobs:
            name, arguments = self._jobs[job_id]
            request_id = request_id or arguments.get('request_id')
        refs = [request_id, result_observation_id]
        duplicate = request_id in self._finished if request_id else False
        for ref in refs:
            previous = self.receipts.get(str(ref), {})
            if ref and (previous.get('status') not in _TERMINAL or
                        previous.get('unknown') and not receipt['unknown'] and receipt['status'] in _TERMINAL):
                self.receipts[str(ref)] = receipt
        while len(self.receipts) > 512: del self.receipts[next(iter(self.receipts))]
        if request_id and receipt['status'] in _TERMINAL:
            if duplicate and (is_poll or receipt['status'] in ('succeeded', 'completed')):
                return  # Status reconciliation is not a second execution.
            self._finished.add(request_id)
        if is_poll and not request_id: return
        if name == 'camera_observe':
            # A genuinely new capture changes evidence scope; repeated status polling does not.
            frames = result.get('frames') or []
            capture = result_observation_id or result.get('observation_id') or result.get('observation_path')
            if capture is None and frames:
                identities = [(f.get('observation_path'),f.get('camera_serial'),f.get('sequence'),
                               f.get('captured_at'),f.get('instance_id'),f.get('connection_epoch'),f.get('source_stamp_s'))
                              for f in frames if isinstance(f,dict)]
                if any(any(x is not None for x in row) for row in identities):
                    capture = _digest(identities)
            if capture is not None: self.view = capture
        view = observation_id or arguments.get('observation_id') or arguments.get('observation_path') or self.view
        # IDs are bookkeeping, not a changed target. Observation identity stays in the signature.
        params = {k: v for k, v in arguments.items() if k != 'request_id'}
        signature = _digest([name, params, view])
        row = {'signature': signature, 'evidence': _digest(_evidence(result)),
               'ref': request_id or f'call-{self.sequence}', 'name': name}
        failed = receipt['rejected'] or receipt['status'] == 'failed'
        comparable = name in _READS or failed and not receipt['unknown']
        if comparable:
            recent = list(self.rows)[-8:]
            same = [x for x in recent if x['signature'] == signature and x['evidence'] == row['evidence']]
            if len(same) >= 2:
                code = 'ignored_feedback' if failed else 'repeated_query'
                self._raise(code, signature, [x['ref'] for x in same] + [row['ref']],
                            'Same request and evidence repeated without a changed prerequisite. Reuse the result or state what new ambiguity another query resolves.')
            if (len(recent) >= 3 and recent[-2]['signature'] == signature
                    and recent[-3]['signature'] == recent[-1]['signature'] != signature
                    and recent[-2]['evidence'] == row['evidence']
                    and recent[-3]['evidence'] == recent[-1]['evidence']):
                self._raise('query_cycle', signature, [x['ref'] for x in recent[-3:]] + [row['ref']],
                            'A/B query cycle on unchanged evidence; choose a new testable hypothesis.')
        for issue in self.issues.values():
            if (issue['state'] == 'awaiting_verification' and issue['code'] in ('repeated_query', 'ignored_feedback', 'query_cycle')
                    and signature != issue['signature'] and not receipt['unknown'] and not receipt['rejected']
                    and (name in _READS and result.get('status') == 'ok' or request_id and receipt['status'] == 'succeeded')):
                issue['state'] = 'resolved'  # Changed observable behavior, not a semantic-success claim.
        self.rows.append(row)
        if request_id and receipt['status'] == 'succeeded':
            self.view = result_observation_id or f'after:{request_id}'
            self.rows.clear()
        if name == 'task_plan' and result.get('task_id') and (
                arguments.get('operation') in ('create','resume') or self.task_id in (None,result['task_id'])):
            self.record_plan(result)
        # A later successful recovery can support a new completion claim, but a
        # bad claim is not silently rewritten by the monitor.

    def snapshot(self):
        active = [copy.deepcopy(i) for i in self.issues.values() if i['state'] != 'resolved']
        state = ('needs_independent_review' if any(i.get('suggest_independent_review') for i in active)
                 else 'needs_review' if any(i['state'] == 'needs_review' for i in active) else 'verifying' if active else 'working')
        return {'state': state, 'task_id': self.task_id, 'issues': active[-8:],
                'scope': 'Public model-tool behavior diagnostics only; not internal adapter computations, task success, private reasoning, a motion gate or an instruction to replay actions.'}
