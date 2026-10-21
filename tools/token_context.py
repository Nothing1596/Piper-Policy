"""Token-budgeted public working context; archived evidence stays retrievable.

Experiment adapter only: no robot calls, provider summaries or action replay.
The counter tokenizes the native chat template; image reservations are explicit
upper bounds, checked against actual prompt usage before returning a decision.
"""
import base64
import copy
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image
from session_context import SessionContext, CONTEXT_TOOL


class ContextBudgetError(ValueError):
    pass


class TokenContext:
    def __init__(self, root, *, context_tokens=32768, output_tokens=8192,
                 safety_tokens=1024, image_tokens=1536):
        self.workspace = Path(root)
        self.workspace.mkdir(exist_ok=False)
        self.cache = SessionContext(self.workspace)
        self.limit = context_tokens - output_tokens - safety_tokens
        if self.limit <= 0 or image_tokens <= 0:
            raise ValueError('Invalid context reservations')
        self.context_tokens = context_tokens
        self.output_tokens = output_tokens
        self.safety_tokens = safety_tokens
        self.image_tokens = image_tokens
        self.turn = 0
        self.memory_rows = {}
        self.followups = []
        self.pending_blocked = None
        self.tool = copy.deepcopy(CONTEXT_TOOL)
        self.tool['description'] += (
            ' Original full images can be read using crop with roi=[0,0,width,height] '
            'from the cached image metadata. read uses start/limit to page exact JSON '
            'records (at most two 4096-character chunks per reply); follow next_start '
            'for the rest. Cached evidence never becomes a live capture.')

    def archive(self, payload):
        messages = copy.deepcopy(payload['messages'])
        images = []
        for mi, message in enumerate(messages):
            if not isinstance(message.get('content'), list):
                continue
            label = ''
            parts = []
            for part in message['content']:
                if part.get('type') == 'text':
                    label = part.get('text', '')
                if part.get('type') != 'image_url':
                    parts.append(part)
                    continue
                url = part['image_url']['url']
                if not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Only delivered image bytes may enter the cache')
                mime, encoded = url.split(';base64,', 1)
                raw = base64.b64decode(encoded, validate=True)
                digest = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as im:
                    width, height = im.size
                identity = hashlib.sha256((label+digest).encode()).hexdigest()[:24]
                path = self.workspace / ('image-'+identity+('.png' if 'png' in mime else '.jpg'))
                if not path.exists():
                    path.write_bytes(raw)
                item = dict(path=str(path), sha256=digest, label=label, width=width,
                            height=height, image_id=path.name, message_index=mi)
                images.append((item, copy.deepcopy(part)))
                parts.append({'type':'text', 'text':'[Cached image '+path.name+']'})
            message['content'] = parts
        current = next((m.get('content') for m in reversed(messages)
                        if isinstance(m.get('content'), str)
                        and m['content'].startswith('CURRENT execution_progress: ')), None)
        self.cache.archive(self.turn, messages, [x[0] for x in images], payload['tools'],
                           {'turn':self.turn, 'execution_progress':current})
        return messages, images

    @staticmethod
    def groups(messages):
        starts = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
        prefix = set(range(starts[0] if starts else len(messages)))
        groups = [set(range(a, b)) for a, b in zip(starts, starts[1:]+[len(messages)])]
        return prefix, groups

    async def pack(self, payload, count_text):
        messages, images = self.archive(payload)
        frame_details = {}
        # Keep the complete per-frame CV metadata in the lossless archive, but
        # project it into the working context only with its selected pixels.
        # This is field selection, not a model-written or semantic summary.
        for message in messages:
            if not isinstance(message.get('content'),list):
                continue
            for part in message['content']:
                if part.get('type')!='text':
                    continue
                try:
                    value=json.loads(part.get('text',''))
                except (ValueError,TypeError):
                    continue
                demo=value.get('demo') if isinstance(value,dict) else None
                if isinstance(demo,dict) and isinstance(demo.get('frames'),list):
                    for frame in demo['frames']:
                        if isinstance(frame,dict) and frame.get('frame_id'):
                            frame_details[frame['frame_id']]=frame
                    demo['frames']=[{k:f[k] for k in ('frame_id','timestamp_s') if k in f}
                                    for f in demo['frames'] if isinstance(f,dict)]
                    demo['frame_details_scope']='Exact CV details accompany selected pixels; all originals remain in session_context.'
                    value.pop('current_plan',None)  # authoritative version is below
                    part['text']=json.dumps(value,ensure_ascii=False)
        progress_views = []
        for message in messages:
            raw=message.get('content')
            if isinstance(raw,str) and raw.startswith('CURRENT execution_progress: '):
                value=json.loads(raw.split(': ',1)[1])
                review=value.get('video_plan_review')
                if isinstance(review,dict) and 'delivered_demo_frames' in review:
                    review['available_cached_demo_frames']=review.pop('delivered_demo_frames')
                progress_views.append((message,value))
                message['content']='CURRENT execution_progress: '+json.dumps(value,ensure_ascii=False)
        prefix, groups = self.groups(messages)
        current_images = [x for x in images if x[0]['label'].startswith('CURRENT simulation evidence')]
        latest_label = current_images[-1][0]['label'] if current_images else None
        # Initial camera state is not a permanent instruction. Deliver the
        # latest capture separately, and label older captures as historical.
        capture_indices = set()
        for i,message in enumerate(messages):
            if not isinstance(message.get('content'),list):
                continue
            for part in message['content']:
                text=part.get('text','')
                if text.startswith('CURRENT simulation evidence'):
                    capture_indices.add(i)
                    if text!=latest_label:
                        part['text']='HISTORICAL prior capture (not current execution authority): '+text
        prefix -= capture_indices
        # The latest action/result group and authoritative progress cannot be
        # replaced by a summary. Older complete groups compete for spare tokens.
        current_indices = {i for i,m in enumerate(messages)
                           if isinstance(m.get('content'),str)
                           and m['content'].startswith('CURRENT execution_progress: ')}
        selected = prefix | current_indices | (groups[-1] if groups else set())
        retrieved = [x for x in images if x[0]['label'].startswith('HISTORICAL CACHE CROP')]
        chosen = []
        for sequence in (current_images[-1:], retrieved[-1:]):
            for item in sequence:
                if item not in chosen:
                    chosen.append(item)
        # Probe callers may provide one unlabelled current image.
        if not chosen and images and not groups and len(images) == 1:
            chosen = images[:1]
        catalog = [{'image_id': x[0]['image_id'], 'label': x[0]['label'][:350],
                    'width': x[0]['width'], 'height': x[0]['height']} for x in images]
        hint = {'role':'user','content':
            'PUBLIC SESSION CACHE: old evidence remains available through session_context list/read/crop. '
            'Use read(turn=N) for exact earlier text, image IDs and actual results; '
            'use crop(image_id=ID,roi=[0,0,width,height]) to view an original full image. '
            'All retrieved images are HISTORICAL; only the current execution_progress observation ID '
            'can authorize a new action. No old action is replayed. '
            'Missing pictures are archived, not negative visual evidence. '+json.dumps(
                {'current_turn':self.turn, 'images_in_source':catalog}, ensure_ascii=False)}
        memo = {}

        async def build(indices, picture_set):
            result = [copy.deepcopy(m) for i,m in enumerate(messages) if i in indices]
            result.append(hint)
            result.append({'role':'user','content':'COMPRESSED PUBLIC MEMORY: '+json.dumps(memory,ensure_ascii=False)})
            if picture_set:
                parts = []
                for item, picture in picture_set:
                    ids=re.findall(r'\bf\d{6}\b',item['label'])
                    cv=next((frame_details[k] for k in ids if k in frame_details),None)
                    label=item['label']+' CACHE image_id='+item['image_id']
                    if item['label'].startswith('CURRENT simulation evidence') and item['label']!=latest_label:
                        label='HISTORICAL prior capture (not current execution authority): '+label
                    if cv is not None:
                        label+=' Exact frame CV: '+json.dumps(cv,ensure_ascii=False)
                    parts += [{'type':'text','text':label}, picture]
                result.append({'role':'user','content':parts})
            stripped = copy.deepcopy(result)
            for m in stripped:
                if isinstance(m.get('content'), list):
                    m['content'] = [p for p in m['content'] if p.get('type') != 'image_url']
            key = json.dumps(stripped, sort_keys=True, ensure_ascii=False)
            if key not in memo:
                count = await count_text(stripped, payload['tools'])
                if type(count) is not int or count < 0:
                    raise ContextBudgetError('Native text token count unavailable')
                memo[key] = count
            bound = memo[key] + len(picture_set)*self.image_tokens
            return result, bound, memo[key]

        memory = self.memory_snapshot()
        packed, bound, text_tokens = await build(selected, chosen)
        omitted_memory = 0
        for key in ('measurements','other_feedback','motion_receipts','blocked_followups'):
            while bound>self.limit and len(memory[key])>1:
                memory[key].pop(0);omitted_memory+=1
                packed,bound,text_tokens=await build(selected,chosen)
        omitted_metadata = 0
        # Accumulated measurement/checkpoint lists are historical data, not
        # authoritative plan state or the latest action receipt. Evict oldest
        # entries only as necessary; the exact originals are already archived.
        for message,progress in progress_views:
            review=progress.get('video_plan_review') or {}
            for entries in (review.get('measurement_index'),progress.get('historical_measurements'),
                            review.get('execution_checkpoints')):
                if not isinstance(entries,list):
                    continue
                while bound>self.limit and len(entries)>1:
                    entries.pop(0);omitted_metadata+=1
                    message['content']='CURRENT execution_progress: '+json.dumps(progress,ensure_ascii=False)
                    packed,bound,text_tokens=await build(selected,chosen)
        if bound > self.limit:
            raise ContextBudgetError('Required instructions, latest feedback and images exceed input budget; archived without dispatch')
        # Allocate at most one third of remaining space to recent complete
        # history first, then fill with evenly spread demonstration frames.
        history_allowance = min(4096, (self.limit-bound)//3)
        initial_bound = bound
        for group in reversed(groups[:-1]):
            candidate, estimate, text = await build(selected | group, chosen)
            if estimate > self.limit or estimate-initial_bound > history_allowance:
                break
            selected |= group
            packed, bound, text_tokens = candidate, estimate, text
        optional = [x for x in images if x not in chosen]
        demos = [x for x in optional if not x[0]['label'].startswith(('CURRENT simulation evidence', 'CANDIDATE', 'HISTORICAL'))]
        # Endpoint then recursive midpoint order preserves coverage as the
        # available token budget changes; no visual downsampling or re-encoding.
        order = []
        if demos:
            order = [0] if len(demos)==1 else [0,len(demos)-1]
            ranges = [(0,len(demos)-1)]
            while ranges:
                a,b = ranges.pop(0)
                if b-a > 1:
                    mid = (a+b)//2
                    order.append(mid); ranges += [(a,mid),(mid,b)]
        additions = [x for x in optional if x not in demos][-1:] + [demos[i] for i in order]
        for picture in additions:
            candidate, estimate, text = await build(selected, chosen+[picture])
            if estimate <= self.limit:
                chosen.append(picture)
                packed, bound, text_tokens = candidate, estimate, text
        # Spare text capacity can retain more recent history without evicting
        # already selected evidence or splitting tool-call/result groups.
        for group in reversed(groups[:-1]):
            if group <= selected:
                continue
            candidate, estimate, text = await build(selected | group, chosen)
            if estimate > self.limit:
                break
            selected |= group
            packed, bound, text_tokens = candidate, estimate, text
        audit = dict(policy='native-token-budget-v2', turn=self.turn,
                     context_tokens=self.context_tokens, output_reserved=self.output_tokens,
                     safety_reserved=self.safety_tokens, input_limit=self.limit,
                     template_text_tokens=text_tokens, image_token_bound=self.image_tokens,
                     prompt_token_bound=bound, delivered_image_ids=[x[0]['image_id'] for x in chosen],
                     cached_image_ids=[x[0]['image_id'] for x in images],
                     omitted_historical_metadata_rows=omitted_metadata,
                     compression_policy=memory['policy'],
                     compressed_memory_rows={k:len(memory[k]) for k in
                         ('motion_receipts','measurements','other_feedback','blocked_followups')},
                     compressed_memory_rows_omitted=omitted_memory,
                     kept_message_indices=sorted(selected), source_messages=len(messages),
                     history_tool_pairs_preserved=True)
        self.cache.publish(self.cache.current_path, {'budget':audit, 'execution_progress':
            next((m['content'] for m in reversed(messages) if m.get('content','') and
                  isinstance(m['content'],str) and m['content'].startswith('CURRENT execution_progress: ')), None)})
        return packed, audit

    def query(self, body):
        result, pixels = self.cache.query(body)
        if body['operation'] != 'read':
            return result, pixels
        # A whole historical input can itself fill 32k. Page exact serialized
        # records instead of injecting that entire conversation a second time.
        records = []
        entries = ([('decision',result['decision'])] if 'decision' in result else [])
        entries += [('actual_tool_result',x) for x in result.get('actual_tool_results',[])]
        entries += [('image',x) for x in result.get('images',[])]
        entries += [('message',x) for x in result['messages']]
        for index,(kind,value) in enumerate(entries):
            raw = json.dumps(value,ensure_ascii=False)
            for offset in range(0,len(raw),4096):
                records.append({'kind':kind,'record':index,'offset':offset,
                                'total_characters':len(raw),'json_text':raw[offset:offset+4096]})
        start=body.get('start',0);limit=min(body.get('limit',2),2)
        return {'session_id':self.cache.session_id,'turn':body['turn'],
                'scope':'HISTORICAL exact JSON chunks; not current execution authority',
                'records':records[start:start+limit],'total':len(records),
                'next_start':start+limit if start+limit<len(records) else None}, []

    def record_response(self, data):
        m = data['choices'][0]['message']
        self.cache.record_decision(self.turn, {k:m[k] for k in ('content','tool_calls') if k in m})
        self.turn += 1

    def record_tool_result(self, turn, value):
        self.cache.record_tool_result(turn, value)
        result=value.get('result') or {}
        args=value.get('arguments') or {}
        if not isinstance(result,dict):
            return
        if not isinstance(args,dict):
            args={}  # Parser failures still have useful, non-execution feedback.
        tool=value.get('name')
        error=result.get('error') or {}
        if not isinstance(error,dict):error={'message':str(error)}
        after=result.get('measured_after',result if tool=='robot_status' else {})
        after=after if isinstance(after,dict) else {}
        robot=after.get('robot',after);robot=robot if isinstance(robot,dict) else {}
        stop=result.get('stop_result');stop=stop if isinstance(stop,dict) else {}
        motion=tool in ('move_to','move_by','move_linear','rotate','robot_move_joints','set_gripper','robot_gripper','robot_stop')
        row={'tool':tool,'source_turn':turn,'status':result.get('status'),
             'request_id':result.get('request_id') or args.get('request_id'),
             'observation_id':result.get('observation_id') or result.get('source_observation_id') or args.get('observation_id') or args.get('observation_path'),
             'error_code':error.get('code') or result.get('error_code'),
             'error_message':str(error.get('message',''))[:240],
             'confirmed_blocked':result.get('confirmed_blocked',False),
             'executed':error.get('executed'),
             'arguments':{k:args[k] for k in ('xyz_m','delta_m','rpy_deg','width_m','effort_n','completion','object_roi','reference_roi') if k in args},
             'evidence':{k:result[k] for k in ('source_measurement_id','point_m','poses','observed_width_m',
                           'observed_height_m','opening_width_m','closing_axis_yaw_deg','geometry_consistency') if k in result},
             'measured_after':{'tcp':after.get('tcp'), 'robot':{k:v for k,v in robot.items()
                               if k in ('q_deg','gripper_width_m','gripper_contact','velocity_deg_s')}},
             'plan_version':result.get('version') if tool=='task_plan' else None,
             'confirmed_stopped':stop.get('confirmed_stopped')}
        # Different outcomes must not collapse just because the arguments are
        # equal. The full original records remain accessible by source turn.
        comparable={k:v for k,v in row.items() if k!='source_turn'}
        identity=hashlib.sha256(json.dumps([tool,args,comparable,
            turn if motion and not row['request_id'] else None],sort_keys=True).encode()).hexdigest()
        previous=self.memory_rows.pop(identity,None)
        row['occurrences']=(previous or {}).get('occurrences',0)+1
        row['first_source_turn']=(previous or {}).get('first_source_turn',turn)
        row['category']='motion_receipts' if motion else 'measurements' if row['evidence'] else 'other_feedback'
        self.memory_rows[identity]=row
        while len(self.memory_rows)>64:self.memory_rows.pop(next(iter(self.memory_rows)))
        if motion and row['status'] in ('succeeded','failed','cancelled','completed'):
            if self.pending_blocked and self.pending_blocked['request_id']!=row['request_id']:
                self.followups.append({'blocked_request':self.pending_blocked['request_id'],
                    'blocked_turn':self.pending_blocked['source_turn'],'next_request':row['request_id'],
                    'next_turn':turn,'next_target':row['arguments'],'next_result':row['status'],
                    'scope':'Subsequent motion only; does not certify recovery of the same subgoal'})
                self.followups=self.followups[-8:]
            self.pending_blocked=row if row['confirmed_blocked'] else None
        self.cache.publish(self.cache.root/'memory.json',self.memory_snapshot())

    def memory_snapshot(self):
        rows=list(self.memory_rows.values())
        return {'policy':'structured-public-memory-v1',
                'scope':'Deterministic receipt/measurement compression, not a semantic success judge. '
                        'Current plan is model-authored; succeeded primitives do not prove completed task steps. '
                        'All measurements retain historical provenance; use current observation eligibility. '
                        'Exact records and omitted history remain in session_context read(source_turn).',
                'motion_receipts':[copy.deepcopy(x) for x in rows if x['category']=='motion_receipts'][-8:],
                'measurements':[copy.deepcopy(x) for x in rows if x['category']=='measurements'][-4:],
                'other_feedback':[copy.deepcopy(x) for x in rows if x['category']=='other_feedback'][-4:],
                'blocked_followups':copy.deepcopy(self.followups[-4:])}
