"""Audited CLI transport; fresh per decision by default, bounded reuse opt-in.

No fabricated provider usage, no action queue, no GT or robot connection in
the decision workspace. Public task memory is owned by the harness, not agy history.
"""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

from session_context import SessionContext, CONTEXT_TOOL


class DecisionRejected(ValueError):
    """Known unexecuted model response; end this case, not the other condition."""


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def usage_delta(current, previous):
    keys = ('input_tokens', 'output_tokens', 'thinking_tokens', 'cache_read_tokens', 'total_tokens')
    if not isinstance(current, dict) or any(type(current.get(k)) is not int or current[k] < 0 for k in keys):
        raise ValueError('agy usage unavailable/malformed; pause, never invent token values')
    delta = {k: current[k] - previous.get(k, 0) for k in keys}
    if any(v < 0 for v in delta.values()):
        raise ValueError('agy cumulative usage regressed')
    if delta['thinking_tokens'] > delta['output_tokens']:
        raise ValueError('agy thinking/output accounting inconsistent')
    if delta['output_tokens'] == 0:
        raise ValueError('agy output usage did not advance; pause, never record a free decision')
    return delta


class AgyDecider:
    def __init__(self, root, model='gemini-3.8-flash-high', executable='agy', exploration=False,
                 provider_turn_window=1):
        if type(provider_turn_window) is not int or not 1 <= provider_turn_window <= 8:
            raise ValueError('provider_turn_window must be an integer in [1, 8]')
        self.provider_turn_window = provider_turn_window
        self.segment_start_turn = 0
        self.provider_tools_digest = None
        self.exploration = exploration
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.workspace = self.root/'public'
        self.workspace.mkdir()
        (self.workspace/'images').mkdir()
        (self.workspace/'.agents').mkdir()
        self.context = SessionContext(self.workspace)
        self.context_tool = CONTEXT_TOOL
        self.conversations = []
        self.state_path = self.root/'guard-state.json'
        self.audit_path = self.root/'guard-events.jsonl'
        self.audit_path.touch()
        self.model, self.executable = model, executable
        self.proc = None
        self.conversation = None
        self.previous_usage = {}
        self.previous_turns = 0
        self.turn = 0
        self.seen_images = set()
        self.prior_message_count = 0
        self.images = {}
        self.image_paths = {}
        self.stderr = None
        hook = Path(__file__).with_name('agy_observation_guard.py').resolve()
        command = shlex.join([sys.executable, str(hook)])
        hooks = {'piper-evidence-only': {
            'PreToolUse': [{'matcher': '*', 'hooks': [{'command': command+' pre '+shlex.quote(str(self.state_path)), 'timeout': 10}]}],
            'PostToolUse': [{'matcher': '*', 'hooks': [{'command': command+' post '+shlex.quote(str(self.state_path)), 'timeout': 10}]}],
            'PreInvocation': [{'command': command+' invocation '+shlex.quote(str(self.state_path)), 'timeout': 10}],
        }}
        save(self.workspace/'.agents/hooks.json', hooks)
        save(self.state_path, {'turn':0, 'images':{}, 'audit':str(self.audit_path)})
        self.schema = {'type':'object', 'additionalProperties':False,
                       'required':['content','plan','tool_calls'], 'properties':{
            'content': {'type':'string'},
            'plan': {'type':'string','minLength':1,'maxLength':2000},
            'tool_calls': {'type':'array', 'minItems':1, 'maxItems':1, 'items':{
                'type':'object','additionalProperties':False,'required':['name','arguments'],
                'properties':{'name':{'type':'string'},'arguments':{'type':'object'}}}}}}
        if exploration:
            self.schema['properties']['tool_calls'].pop('maxItems')
        save(self.root/'response-schema.json', self.schema)

    async def start(self):
        if self.proc:
            return
        self.conversation = None
        self.previous_usage = {}
        self.previous_turns = 0
        self.segment_start_turn = self.turn
        self.stderr = (self.root/'agy-stderr.log').open('ab')
        self.proc = await asyncio.create_subprocess_exec(
            self.executable, '--model', self.model, '--disable-slash-commands',
            '--input-format','stream-json','--output-format','stream-json',
            '--json-schema',str(self.root/'response-schema.json'),
            cwd=self.workspace, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=self.stderr,
            limit=16*1024*1024,
            env={k:v for k,v in os.environ.items() if k not in ('DEEPSEEK_API_KEY','ANTHROPIC_API_KEY','OPENAI_API_KEY')})

    async def close(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), 8)
            except asyncio.TimeoutError:
                self.proc.terminate()
                try: await asyncio.wait_for(self.proc.wait(), 5)
                except asyncio.TimeoutError:
                    self.proc.kill(); await self.proc.wait()
            self.proc = None
        if self.stderr:
            self.stderr.close(); self.stderr = None

    def prepare(self, payload):
        # The CLI's native evidence reader is not a returned harness action.
        # Publish the actual tool names to its structured-output contract too.
        names = [t['function']['name'] for t in payload['tools']]
        self.schema['properties']['tool_calls']['items']['properties']['name']['enum'] = names
        save(self.root/'response-schema.json', self.schema)
        messages = payload['messages']
        rendered = []
        all_rendered = []
        manifest = []
        for index, message in enumerate(messages):
            content = message.get('content')
            if isinstance(content, list):
                parts = []
                label = ''
                for part in content:
                    if part['type'] == 'text':
                        label = part['text']; parts.append(label)
                    elif part['type'] == 'image_url':
                        url = part['image_url']['url']
                        if not url.startswith(('data:image/jpeg;base64,','data:image/png;base64,')):
                            raise ValueError('Expected actual image bytes, not path/placeholder')
                        raw = base64.b64decode(url.split(',',1)[1],validate=True)
                        digest = hashlib.sha256(raw).hexdigest()
                        if digest not in self.image_paths:
                            self.image_paths[digest] = self.workspace/('frame-%03d' % (len(self.image_paths)+1)
                                +('.png' if url.startswith('data:image/png') else '.jpg'))
                        path = self.image_paths[digest]
                        if not path.exists():path.write_bytes(raw)
                        self.images[str(path)] = digest
                        manifest.append({'path':str(path),'sha256':digest,'label':label,'message_index':index})
                        parts.append('IMAGE '+path.name+'; exact view_file arguments: '+json.dumps({'AbsolutePath':str(path)}))
                    else:raise ValueError('Unexpected multimodal part')
                text = '\n'.join(parts)
            else:text = str(content or '')
            row = {'role': message['role'], 'content': text}
            if message.get('tool_calls'): row['tool_calls'] = message['tool_calls']
            if message.get('tool_call_id'): row['tool_call_id'] = message['tool_call_id']
            all_rendered.append(row)
            if self.turn == 0 or index >= self.prior_message_count:
                rendered.append(row)
        required = {m['path'] for m in manifest if m['path'] not in self.seen_images}
        # A fresh provider segment sees the latest capture again, not an old summary.
        latest_live = max((m['message_index'] for m in manifest
                           if m['label'].startswith('CURRENT simulation evidence')), default=-1)
        for m in manifest:
            if m['message_index'] == latest_live:
                required.add(m['path'])
        save(self.root/f'turn-{self.turn:03d}-images.json',manifest)
        # execution_progress is appended transiently by runner; do not skip next permanent message.
        self.next_message_count = len(messages)-1
        instructions = (
            'You are the decision model in an isolated robot SIMULATION experiment. '
            'The controller executes exactly one returned tool call, then sends actual feedback before your next decision. '
            'Use view_file to SEE every image in REQUIRED_IMAGE_VIEWS before answering; actual pixels, not filenames. '
            'view_file is a NATIVE evidence-reading call: invoke it now, never put it in the returned tool_calls array. '
            'The returned tool_calls array may contain ONLY names from AVAILABLE_ROBOT_TOOLS. '
            'Public session cache files and exported images may be read with view_file; all historic exports stay available. '
            'Shell, code, networking, private answers, other sessions and delegation are blocked. '
            'Do not write or queue scripts/actions. Return one JSON tool decision matching the supplied tool schema; '
            'do not call native robot tools yourself. No private reasoning is requested. '
            'Copy exact AbsolutePath values. If context was compacted, read context/current.json, context/tools.json and context/index.json at the absolute paths below; never guess paths or use private transcripts. '
            'The final response MUST be exactly {"content":"brief observation and image IDs supporting this decision", '
            '"plan":"short current task plan", "tool_calls":[{"name":"chosen_tool", "arguments":{...}}]}. '
            'Your plan is persisted through the existing task_plan store and read back as current_plan on the next turn. '
            'On the first decision state the demonstration-derived steps; thereafter update observed progress, next step, '
            'and relevant uncertainty. Keep it short and revise it when feedback disagrees. This is a provisional guide, '
            'not a requirement to hit estimated numeric targets. No extra task_plan tool call is needed for this writeback. '
            'In content, cite frame IDs that changed or supported your decision; if a view is occluded or adds nothing, say so. '
            'Before finish_task with outcome=completed, actually open/release, move the gripper away by at least 0.03 m using a justified path, '
            'and inspect fresh feedback. Final acceptance also checks no robot-object contact and stability over 2 seconds. '
            'For an unsuccessful attempt use finish_task with outcome=failed or aborted; do not make extra motions to satisfy success checks. '
            'Do not return a bare name/arguments object or markdown fences. '
            'Derive the task from this session demonstration, never remembered benchmark answers. A fresh CLI segment continues the SAME task; do not restart completed actions. '
            'Retain uncertainty and use subsequent feedback. Source labels distinguish demo and live images. '
        )
        if self.exploration:
            instructions = instructions.replace('exactly one returned tool call', 'the returned tool calls concurrently where independent')
            instructions = instructions.replace('Return one JSON tool decision', 'Return JSON tool decisions; batch independent tools when useful')
            instructions += (' All listed Piper CLI model tools are available. There is no cumulative token cutoff. '
                'State-changing calls are coordinated for the single robot; reads can overlap. '
                'Do not assume a batch is atomic or that dependent calls have seen each other’s results. '
                'Grip-state changes invalidate stale actions and return new images; revise your plan from that feedback. ')
        # Deterministic extraction, not an LLM-written summary. Full deltas stay on disk.
        hot_indices = {i for i in range(len(messages)) if i >= self.prior_message_count}
        hot_indices.update(i for i,m in enumerate(messages) if m.get('role') == 'system')
        if latest_live >= 0: hot_indices.add(latest_live)
        if len(messages)>1: hot_indices.add(1)  # Original task, never inferred from plan.
        hot = []
        for i in sorted(hot_indices):
            row = dict(all_rendered[i])
            if self.turn and i == 1 and isinstance(messages[i].get('content'), list):
                texts = [p['text'] for p in messages[i]['content'] if p['type']=='text']
                row['content'] = texts[0] if texts else 'Original demonstration is in the session cache.'
            hot.append(row)
        recent = []
        for number in range(max(0,self.turn-2),self.turn):
            entry = self.context.turns.get(number,{})
            if entry.get('decision_path'):
                recent.append(self.context.read_public(entry['decision_path']))
        capsule = {'session_id':self.context.session_id, 'turn':self.turn,
                   'numbering':'zero-based', 'public_messages':hot, 'recent_model_candidates':recent,
                   'required_image_views':sorted(required),
                   'scope':'Candidates are not outcomes. Current execution_progress and actual feedback take precedence. Historical views never refresh motion authority.'}
        self.context.archive(self.turn,rendered,manifest,payload['tools'],capsule)
        self.sync_guard()
        instructions += (' Each robot decision uses a fresh agy conversation; this harness session persists on disk. '
            'Use session_context list/read to retrieve a prior turn, view_file to inspect its exact image paths, '
            'or session_context crop to process historical pixels. Historical observations cannot authorize current motion. ')
        if self.provider_turn_window > 1:
            instructions = instructions.replace(
                'Each robot decision uses a fresh agy conversation;',
                f'A provider conversation is reused for at most {self.provider_turn_window} decisions, then refreshed;')
        prompt = instructions+'\nSESSION_CACHE (exact view_file AbsolutePath values):\n'+json.dumps(self.cache_paths())
        prompt += '\nREQUIRED_IMAGE_VIEWS (copy exact paths):\n'+json.dumps(sorted(required))
        prompt += '\nWORKING_CONTEXT:\n'+json.dumps(capsule,ensure_ascii=False)
        if not self.exploration:
            prompt+='\nRemaining total output budget INCLUDING thinking: '+str(payload.get('max_tokens'))+' tokens. Make a concise next decision, not lengthy analysis.'
        prompt+='\nAVAILABLE_ROBOT_TOOLS:\n'+json.dumps(payload['tools'],ensure_ascii=False)
        return prompt, required

    def cache_paths(self):
        return {name:str(path) for name,path in [('current',self.context.current_path),
                ('tools',self.context.contract_path),('index',self.context.index_path)]}

    def sync_guard(self):
        save(self.state_path, {'turn':self.turn,'images':self.context.files,
                              'recovery':self.cache_paths(),'audit':str(self.audit_path)})

    def context_call(self, body):
        return self.context.query(body)

    def record_tool_result(self, turn, value):
        self.context.record_tool_result(turn,value)

    async def request(self, prompt):
        await self.start()
        self.proc.stdin.write((json.dumps({'event':'user','message':{'content':prompt}},ensure_ascii=False)+'\n').encode())
        await self.proc.stdin.drain()
        with (self.root/f'turn-{self.turn:03d}-stream.jsonl').open('wb') as out:
            while True:
                line = await self.proc.stdout.readline()
                if not line:raise RuntimeError('agy closed before a result; no robot request dispatched')
                out.write(line);out.flush()
                data = json.loads(line)
                if data.get('event') == 'init':
                    ident = data['conversation_id']
                    if self.conversation is not None:raise ValueError('agy repeated initialization or session changed')
                    if ident in self.conversations:raise ValueError('Provider reused a prior decision session')
                    self.conversation = ident
                    self.conversations.append(ident)
                if data.get('event') == 'result':return data['result']

    async def __call__(self, args, client, payload, ident):
        try:
            return await self._decide(args, client, payload, ident)
        except BaseException:
            await self.close()
            raise
        finally:
            # Inference lifetime only; no robot request is replayed or closed here.
            if self.provider_turn_window == 1 or self.turn-self.segment_start_turn >= self.provider_turn_window:
                await self.close()

    async def _decide(self, args, client, payload, ident):
        if self.proc and self.proc.returncode is not None:
            # Only refresh a process known to have exited BEFORE this request.
            # A failure after dispatch remains an error, never an automatic retry.
            await self.close()
        tools_digest=hashlib.sha256(json.dumps(payload['tools'],sort_keys=True).encode()).hexdigest()
        if self.proc and tools_digest != self.provider_tools_digest:
            # The provider loads its response schema when starting. A changed
            # catalog must never continue with an old structured-output schema.
            await self.close()
        self.provider_tools_digest=tools_digest
        prompt, required = self.prepare(payload)
        (self.root/f'turn-{self.turn:03d}-prompt.txt').write_text(prompt)
        started = time.time()
        try:
            result = await asyncio.wait_for(self.request(prompt),None if self.exploration else 300)
        except BaseException as exc:
            save(self.root/f'turn-{self.turn:03d}-transport-error.json',
                 {'request_id':ident,'requested_model':self.model,'started':started,'ended':time.time(),
                  'error_type':type(exc).__name__,'usage':None,'cost':None,'action_dispatched':False})
            raise
        save(self.root/f'turn-{self.turn:03d}-result.json',result)
        if result.get('status') != 'SUCCESS':raise RuntimeError('agy did not succeed: '+str(result.get('error',result.get('status'))))
        if result.get('conversation_id') != self.conversation:raise ValueError('agy result session mismatch')
        if type(result.get('num_turns')) is not int or result['num_turns'] <= self.previous_turns:
            raise ValueError('No fresh CLI decision for this turn')
        delta = usage_delta(result.get('usage'),self.previous_usage)
        hook_rows = [json.loads(l) for l in self.audit_path.read_text().splitlines()]
        rows = [r for r in hook_rows if r['turn']==self.turn and r['conversation_id']==self.conversation]
        image_views = [r for r in rows if r['phase']=='post' and r['tool']=='view_file'
                       and r['path'] and Path(r['path']).suffix in ('.jpg','.png') and r['allowed'] and not r.get('error')]
        seen = {r['path'] for r in image_views}
        receipt = dict(request_id=ident, session_id=self.context.session_id,
                       context_policy='harness-managed-per-decision' if self.provider_turn_window==1 else 'harness-managed-bounded-provider-session',
                       provider_turn_window=self.provider_turn_window, segment_start_turn=self.segment_start_turn,
                       conversation_id=self.conversation, requested_model=self.model,
                       runtime_models=sorted({r['model'] for r in rows if r.get('model')}),
                       started=started, ended=time.time(), cumulative_usage=result['usage'], usage_delta=delta,
                       required_images=sorted(required), confirmed_image_views=sorted(seen),
                       image_view_calls=len(image_views),
                       blocked_tool_calls=sum(r['phase']=='pre' and not r['allowed'] for r in rows),
                       model_invocations=sum(r['phase']=='invocation' for r in rows),
                       cli_turns_delta=result['num_turns']-self.previous_turns,
                       provider_cost=None, usage_source='per-conversation cumulative result delta; run totals sum receipts; thinking included in output',
                       tools_sha256=self.context.files[str(self.context.contract_path)],
                       context_sha256=self.context.files[str(self.context.current_path)])
        save(self.root/f'turn-{self.turn:03d}-receipt.json',receipt)
        if required-seen:raise DecisionRejected('Image delivery not confirmed; no robot request dispatched: '+str(len(required-seen)))
        if not self.exploration and delta['output_tokens'] > payload['max_tokens']:raise DecisionRejected('Real CLI output exceeded remaining budget; no robot request dispatched')
        from jsonschema import validate, ValidationError
        try:
            decision = result.get('structured_output')
            if decision is None:decision=json.loads(result['response'])
            validate(decision,self.schema)
        except (ValueError,KeyError,TypeError,ValidationError) as exc:
            raise DecisionRejected('Invalid structured decision; no robot request dispatched') from exc
        calls=decision['tool_calls']
        allowed={t['function']['name'] for t in payload['tools']}
        if any(call['name'] not in allowed for call in calls):raise DecisionRejected('Unexposed robot tool in decision')
        self.context.record_decision(self.turn,decision)
        self.previous_usage=result['usage'];self.previous_turns=result['num_turns']
        self.prior_message_count=self.next_message_count;self.seen_images.update(seen);self.turn+=1
        return {'model':self.model,'usage':{'prompt_tokens':delta['input_tokens'],
                'completion_tokens':delta['output_tokens'],'total_tokens':delta['input_tokens']+delta['output_tokens'],
                'completion_tokens_details':{'reasoning_tokens':delta['thinking_tokens']}},
                'plan_update':decision['plan'],
                'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':decision['content'],
                'tool_calls':[{'id':ident+'-'+str(i),'type':'function','function':{'name':call['name'],
                    'arguments':json.dumps(call['arguments'])}} for i,call in enumerate(calls)]}}]}
