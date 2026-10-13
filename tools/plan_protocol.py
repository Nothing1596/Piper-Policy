"""Small experiment-only protocol; public task_plan and robot tools are unchanged."""
import ast
import json
from longvil_metrics import RELATIONS, semantic_actions


def strict_program_actions(code, objects=None):
    """Prediction formatting gate; GT continues to use semantic_actions."""
    if not isinstance(code,list) or not all(isinstance(line,str) for line in code):
        raise ValueError('code must be a list of strings')
    text='\n'.join(code).replace('<begin_code>','').replace('<end_code>','')
    text='\n'.join(line for line in text.splitlines() if not line.strip().startswith(chr(96)*3))
    if len(text)>200000:raise ValueError('program exceeds analysis budget')
    tree=ast.parse(text);seen=set()
    for node in tree.body:
        if node.lineno in seen:raise ValueError('Multiple statements on one physical line')
        if node.end_lineno!=node.lineno:raise ValueError('Multiline statements are not allowed')
        seen.add(node.lineno)
    if any(';' in line for line in text.splitlines() if not line.strip().startswith('#')):
        raise ValueError('Semicolons / semicolon chains are not allowed')
    return semantic_actions(code,objects)


class Protocol:
    EXPECTED=('get','update','update','get')

    def __init__(self,mode,task_id=None,version=None,objects=()):
        if mode not in ('R0','R1','R2'):raise ValueError('Unknown protocol mode')
        self.mode,self.task_id,self.version=mode,task_id,version
        self.objects=set(objects);self.current_phase=-1;self.delivered_frame_ids=set()
        self.errors=[];self.phase_success=[False]*4;self.phase_tool_calls=[[] for _ in range(4)]
        self.last_read_plan=None;self.last_read_version=None;self.last_updated_plan=None
        self.consistent=None

    @property
    def adherent(self):
        return not self.errors and (self.mode=='R0' or all(self.phase_success))

    def fail(self,code,message):
        self.errors.append(code);raise ValueError(message)

    def begin_phase(self,index,frame_ids=()):
        if index not in range(5):raise ValueError('Unknown phase')
        self.current_phase=index;self.delivered_frame_ids.update(frame_ids)

    def parse_interpretation(self,raw):
        if not isinstance(raw,str):self.fail('invalid_interpretation_format','interpretation must be a JSON string')
        if len(raw)>2000:self.fail('interpretation_too_long','interpretation exceeds 2000 chars')
        try:value=json.loads(raw)
        except (ValueError,TypeError):self.fail('interpretation_json_invalid','interpretation is not valid JSON')
        if not isinstance(value,dict) or set(value)!={'actions','evidence_refs','notes'}:
            self.fail('interpretation_keys_invalid','Exact keys actions, evidence_refs, notes are required')
        actions,refs,notes=value['actions'],value['evidence_refs'],value['notes']
        if not isinstance(notes,str):self.fail('notes_not_str','notes must be a short string within the 2000-character interpretation')
        if not isinstance(actions,list) or not 1<=len(actions)<=128:
            self.fail('actions_count_invalid','actions must contain 1..128 entries')
        if not isinstance(refs,list) or len(refs)!=len(actions):
            self.fail('evidence_refs_length_mismatch','Evidence references must parallel actions')
        for action,citations in zip(actions,refs):
            if not isinstance(action,list) or not all(isinstance(v,str) for v in action):
                self.fail('invalid_action_primitive','Action must be an array of strings')
            if len(action)==4 and action[0]=='pick_place':
                if self.objects and (action[1] not in self.objects or action[2] not in self.objects):
                    self.fail('unknown_object','Unknown object')
                if action[3] not in RELATIONS:self.fail('unknown_relation','Unknown spatial relation')
            elif action not in (['open'],['close']):
                self.fail('invalid_action_primitive','Invalid action primitive')
            if not isinstance(citations,list) or not 1<=len(citations)<=4:
                self.fail('evidence_ref_count_invalid','Each action needs 1..4 frame references')
            if any(not isinstance(fid,str) or fid not in self.delivered_frame_ids for fid in citations):
                self.fail('undelivered_frame_ref','Reference is not an already delivered frame')
        return value

    def before_call(self,name,args):
        if name not in (('submit_program',) if self.mode=='R0' else ('submit_program','task_plan')):
            self.fail('unavailable_tool','Unavailable tool')
        if name=='submit_program':
            if self.current_phase!=4:self.fail('early_submission','Early submission is not allowed')
            return
        op=args.get('operation')
        if op not in ('get','update'):self.fail('unavailable_operation','Only get/update are exposed')
        if self.mode=='R2' and (self.current_phase==4 or op!=self.EXPECTED[self.current_phase]):
            self.fail('wrong_phase','Wrong-phase plan operation')
        if args.get('task_id',self.task_id)!=self.task_id:self.fail('task_id_mismatch','task_id mismatch')
        if op=='update':
            if type(args.get('expected_version')) is not int or args['expected_version']!=self.version:
                self.fail('version_mismatch','expected_version mismatch')
            changes=args.get('changes')
            if not isinstance(changes,dict) or 'interpretation' not in changes:
                self.fail('missing_interpretation','changes must include interpretation')
            self.parse_interpretation(changes['interpretation'])
            for question in changes.get('questions',[]):
                if not isinstance(question,dict) or question.get('source')!='demo':
                    self.fail('invalid_question_source','Question source must be demo')
                if set(question)&{'query','state','result','evidence','answered_at','stale_at'}:
                    self.fail('forbidden_question_keys','Numeric/backend-owned fields cannot be written')

    def after_call(self,name,args,result,is_error=False):
        if is_error:self.errors.append('tool_error');return
        if name!='task_plan':return
        op=args.get('operation');expected=self.version+(op=='update')
        if not isinstance(result,dict) or result.get('task_id')!=self.task_id:
            self.errors.append('get_task_mismatch');return
        if type(result.get('version')) is not int or result['version']!=expected:
            self.errors.append('version_not_advanced_by_1' if op=='update' else 'get_version_mismatch');return
        if op=='update':self.version=expected;self.last_updated_plan=result
        else:self.last_read_plan=result;self.last_read_version=expected
        if self.current_phase<4:self.phase_tool_calls[self.current_phase].append((op,result))

    def finish_phase(self):
        if self.current_phase==4:return
        expected=self.EXPECTED[self.current_phase]
        ok=any(op==expected for op,_ in self.phase_tool_calls[self.current_phase])
        self.phase_success[self.current_phase]=ok
        if self.mode=='R2' and not ok:self.errors.append('missing_r2_step_phase_'+str(self.current_phase))

    def accept_submission(self,body,current_plan=None):
        try:actions=strict_program_actions(body['code'],self.objects)
        except (ValueError,SyntaxError,KeyError,TypeError) as exc:
            self.errors.append('syntax_error: '+str(exc));return False
        plan=current_plan or self.last_updated_plan or self.last_read_plan
        if plan and plan.get('interpretation'):
            try:self.consistent=actions==[tuple(a) for a in self.parse_interpretation(plan['interpretation'])['actions']]
            except ValueError:self.consistent=False
        if self.mode!='R2':return True
        if not self.adherent:self.errors.append('r2_missing_mandatory_phase');return False
        if not plan or plan.get('version')!=self.last_read_version or self.version!=self.last_read_version:
            self.errors.append('r2_final_read_version_mismatch');return False
        if not self.consistent:self.errors.append('r2_plan_code_inconsistent');return False
        return True
