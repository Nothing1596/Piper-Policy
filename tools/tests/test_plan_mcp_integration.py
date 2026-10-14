"""Opt-in real CLI/MCP fixture: no actor API and no robot-action calls."""
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
import pytest
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0,str(Path(__file__).parents[1]))
from plan_protocol import Protocol


@pytest.mark.skipif(os.environ.get('PIPER_PLAN_REAL_MCP')!='1',reason='Explicit real CLI fixture only')
@pytest.mark.asyncio
async def test_real_plan_tools_and_owned_executor_cleanup(tmp_path):
    repo=Path(__file__).parents[2];cli=Path(sys.executable).with_name('piper-robot')
    if not cli.exists():pytest.skip('piper-robot entrypoint absent')
    params=StdioServerParameters(command=str(cli),args=['--root',str(tmp_path),'--mode','simulation',
        '--simulation-backend','sim','--agent-mcp'],env={'PATH':os.environ.get('PATH',''),
        'PYTHONPATH':os.pathsep.join([str(repo/'piperx-cli/src'),str(repo/'piper-lab/src')])})
    calls=[]
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=45)) as session:
            await session.initialize()
            async def call(body):
                calls.append('task_plan');result=await session.call_tool('task_plan',body)
                assert not result.isError
                return result.structuredContent
            state=await call({'operation':'create','goal':'offline integration fixture'})
            protocol=Protocol('R2',state['task_id'],state['version'],['drawer'])
            text=json.dumps({'actions':[['open']],'evidence_refs':[['fixture_f0']],'notes':'fixture evidence'})
            for index,operation in enumerate(['get','update','update','get']):
                protocol.begin_phase(index,['fixture_f0']);body={'operation':operation,'task_id':state['task_id']}
                if operation=='update':body.update(expected_version=state['version'],changes={'interpretation':text,'questions':[]})
                protocol.before_call('task_plan',body);state=await call(body)
                protocol.after_call('task_plan',body,state);protocol.finish_phase()
            protocol.begin_phase(4,['fixture_f0'])
            assert protocol.accept_submission({'code':['open()']},state)
            assert protocol.adherent and state['version']==3
            print(json.dumps({'real_cli_mcp':True,'phase_success':protocol.phase_success,
                             'accepted':True,'robot_action_calls':0,'calls':calls}))
    assert not (tmp_path/'profiles/simulation/local/runtime.json').exists()
