"""Opt-in public video/plan review, with no grading or robot side effects."""
import json
from piperx_middleware.behavior_monitor import completion_issues, public_receipt

VIDEO_PLAN_GUIDANCE = '''Video-to-plan sequence review (sequence-review-v1):
Reproduce the FULL demonstrated action sequence, not just its final arrangement. Before the first object manipulation, review the recorded overview from beginning to end. Use demo_watch to inspect ambiguous transitions or gaps. Preserve temporary stacks, repeated moves of the same object, and changes of reference object. Do not assume an action count or invent hidden actions; mark uncertainty and cite delivered frame IDs/times.
Use existing task_plan.update to save a compact JSON STRING in interpretation (maximum 2000 characters), e.g. {"steps":[{"id":"s1","action":["pick_place","object","reference","top"],"evidence_refs":["DELIVERED_DEMO_FRAME"],"status":"pending","execution_refs":[]}],"notes":"Remaining ambiguity"}. Use only observed object names and actual evidence IDs; use pending/completed/uncertain. Stage names use letters, digits, underscore or hyphen, e.g. execute_review. Read the authoritative current task_id/version from execution_progress; no guessed IDs.
After a release, a blocked move, changed grip or a newly discovered demonstration step, update the ledger with observed progress and remaining/omitted steps before continuing the next semantic transfer. A failed or partial primitive is not a completed pick_place. A completed step needs actual transfer/release evidence (request IDs and live IDs), not just contact or your intention. Retain earlier steps and repeated moves when revising; do not overwrite the sequence with only the final goal. Review each demonstrated step and remaining uncertainty before finish_task. Plans are revisable hypotheses, never exact geometric targets or motion permission; feedback may require a different path or an unsuccessful finish.
Use video_plan_review's public execution checkpoints and measurement index before measuring again. Historical geometry can help remember dimensions or why a decision was made, but does not authorize a current target after movement. Obtain fresh evidence for changed/occluded objects or reference surfaces. If a needed dimension is already established and no relevant change is observed, cite it instead of probing many neighbouring pixels. Prefer one justified next action and its feedback to exhaustive geometry mapping. Dependent batched calls must use actual returned observations, never presumed IDs.
Use harness_review as advisory feedback about repeated public behavior. Explain the changed prerequisite or revise your interpretation before repeating the same query or rejected target. Renaming the stage alone does not address a review issue. It never decides task success, grants motion permission or automatically calls a supervisor.
Write progress from an action in the NEXT decision, after reading its actual outcome, observation and authoritative plan_writeback version. Do not batch a post-action plan update with the action: you have not received that feedback yet. If an update is stale, use the returned current_plan, reconsider the change and submit a new update; never replay the action to fix a plan conflict. Wording changes with the same stable step IDs are allowed.
'''


def review_context(plan, chain, measurements, valid_views, reviewed_length, delivered_frames=()):
    """Review cues from model-written state and public actual tool results only."""
    try:
        ledger=json.loads((plan or {}).get('interpretation',''))
        if not isinstance(ledger,dict) or not isinstance(ledger.get('steps'),list):ledger=None
    except (ValueError,TypeError):ledger=None
    reasons=[];issues=[]
    receipts={str(ref):public_receipt(row.get('actual_result')) for row in chain for ref in
              ((row.get('model_arguments') or {}).get('request_id'),row.get('result_observation_id')) if ref is not None}
    if ledger is None or not ledger['steps']:
        reasons.append('Record the full demonstrated sequence with frame references; current plan has no step ledger.')
    if ledger:
        ids=set()
        for step in ledger['steps']:
            if not isinstance(step,dict):issues.append('Each step must be an object.');continue
            ident=step.get('id')
            if not isinstance(ident,str) or not ident or ident in ids:issues.append('Use unique step IDs, retaining repeated object moves as separate steps.')
            if isinstance(ident,str):ids.add(ident)
            if step.get('status') not in ('pending','completed','uncertain'):issues.append('Step status must be pending/completed/uncertain.')
            refs=step.get('evidence_refs')
            if not isinstance(refs,list) or not refs or any(not isinstance(ref,str) or ref not in delivered_frames for ref in refs):
                issues.append('Cite delivered demo frame IDs for each step; request missing evidence with demo_watch.')
        issues.extend(completion_issues(ledger,receipts))
    checkpoints=[]
    for index,row in enumerate(chain):
        result=row.get('actual_result') or {};body=row.get('model_arguments') or {}
        released=(row.get('tool') in ('set_gripper','robot_gripper')
                  and result.get('status')=='succeeded' and body.get('width_m',0)>=.06)
        blocked=result.get('error_code')=='motion_blocked'
        if released or blocked:
            checkpoints.append({'chain_index':index,'kind':'open_gripper' if released else 'blocked',
                'request_id':body.get('request_id'),'status':result.get('status'),
                'observation_id':row.get('result_observation_id'),
                'scope':'Opening alone is not proof of an object transfer.' if released else 'Use partial measured result, not the original commanded endpoint.'})
            if index>=reviewed_length:
                reason='Review observed releases and remaining steps.' if released else 'Review partial/blocked motion and remaining steps.'
                if reason not in reasons:reasons.append(reason)
    index=[]
    # Deduplicate identical historical queries without inventing cross-frame validity.
    seen=set()
    for m in reversed(measurements):
        key=json.dumps([m.get('source_observation_id'),m.get('tool'),m.get('arguments')],sort_keys=True)
        if key in seen:continue
        seen.add(key)
        evidence=m.get('evidence') or {}
        summary={k:evidence[k] for k in ('source_measurement_id','observed_width_m','observed_height_m',
                 'observed_surface_spans_m','point_m','relative_reference','placement_estimate') if k in evidence}
        if isinstance(evidence.get('surface_summary'),dict):
            summary['surface_summary']={k:evidence['surface_summary'][k] for k in
                ('median_point_m','observed_bounds_m') if k in evidence['surface_summary']}
        index.append({'tool':m['tool'],'source_observation_id':m.get('source_observation_id'),
                      'arguments':m.get('arguments'),'evidence':summary,
                      'matches_current_view':m.get('source_observation_id') in valid_views,
                      'scope':'Saved measurement; historical views are not recertified current targets.'})
        if len(index)>=24:break
    return {'ledger':ledger,'ledger_issues':list(dict.fromkeys(issues)),
            'delivered_demo_frames':list(delivered_frames),
            'review_reasons':reasons,'execution_checkpoints':checkpoints,
            'measurement_index':list(reversed(index)),
            'scope':'Public evidence and model-authored progress; no GT, scores, automatic completion labels or action gates.'}
