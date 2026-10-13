import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from piperx_middleware.agent_console import dispatch


@pytest.mark.asyncio
async def test_model_cannot_approve_configure_or_invent_action_id():
    calls=[]
    class Controller:
        def available_tools(self):return [{'name':'move_to','inputSchema':{'required':['request_id']}}]
        async def invoke_tool(self,name,args):calls.append((name,args));return {'status':'ok'}
    c=Controller()
    for command in ('approve','config','mode','manual','shutdown'):
        with pytest.raises(ValueError):await dispatch(c,{'id':1,'command':command})
    result=await dispatch(c,{'id':2,'command':'call','tool':'move_to','arguments':{}})
    assert result['error']['code']=='request_id_required' and not calls
    await dispatch(c,{'id':3,'command':'call','tool':'move_to','arguments':{'request_id':'same-id'}})
    assert calls==[('move_to',{'request_id':'same-id'})]


@pytest.mark.asyncio
async def test_stop_cancels_action_preparation_before_dispatch():
    dispatched=[]
    async def preparing():
        await asyncio.sleep(10)
        dispatched.append('motion')
    task=asyncio.create_task(preparing())
    class Controller:
        background_task=task
        async def invoke_tool(self,name,args):
            dispatched.append(name)
            return {'status':'cancellation_requested'}
    result=await dispatch(Controller(),{'id':1,'command':'call','tool':'robot_stop','arguments':{}})
    await asyncio.gather(task,return_exceptions=True)
    assert result['status']=='cancellation_requested' and dispatched==['robot_stop']


def test_actual_jsonl_process_actions_errors_and_quit(tmp_path):
    proc=subprocess.Popen([sys.executable,'-m','piperx_middleware.standalone_cli','--root',str(tmp_path),
        '--mode','simulation','--simulation-backend','sim','--agent'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def call(payload):
        proc.stdin.write(json.dumps(payload)+'\n');proc.stdin.flush()
        response=json.loads(proc.stdout.readline())
        assert response['id']==payload['id']
        return response['result']
    try:
        assert json.loads(proc.stdout.readline())['result']['schema']=='piper.agent.v1'
        assert call({'id':'plan','command':'call','tool':'task_plan','arguments':{'operation':'create','goal':'Check JSONL task persistence'}})['task_id']
        assert call({'id':'connect','command':'call','tool':'robot_connect','arguments':{}}).get('error') is None
        assert call({'id':'move','command':'call','tool':'robot_move_joints',
                     'arguments':{'joints_deg':[1.,0.,0.,0.,0.,0.],'request_id':'jsonl-test-action-01'}})['status']=='succeeded'
        state=call({'id':'state','command':'status'})
        assert state['robot']['q_deg'][0]==1. and 'diagnostics' not in state['robot']
        assert call({'id':'blocked','command':'approve'})['error']['code']=='invalid_request'
        assert call({'id':'quit','command':'quit'})['status']=='draining'
        assert proc.wait(timeout=10)==0,proc.stderr.read()
        assert not list(tmp_path.rglob('runtime.json'))
    finally:
        proc.stdin.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=5)


def test_agent_requires_explicit_mode(tmp_path):
    result=subprocess.run([sys.executable,'-m','piperx_middleware.standalone_cli','--root',str(tmp_path),'--agent'],
                          input='',capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'explicit --mode' in result.stderr
