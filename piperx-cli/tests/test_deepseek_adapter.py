import asyncio
import json
import httpx
import pytest
from piperx_middleware import model_agent


@pytest.mark.parametrize('endpoint,extension', [
    ('https://api.deepseek.com', True),
    ('https://api.deepseek.com/v1', True),
    ('https://api.openai.com/v1', False),
    ('http://127.0.0.1:1234/v1', False),
])
def test_probe_and_tool_result_round_trip(monkeypatch, endpoint, extension):
    requests = []
    invoked = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if extension:
            assert body['thinking'] == {'type':'disabled'}
        else:
            assert 'thinking' not in body
        if len(requests) == 2:
            msg = {'role':'assistant','content':None,'tool_calls':[
                {'id':'call_1','type':'function','function':{'name':'robot_status','arguments':'{}'}}]}
        else:
            if len(requests) == 3:
                assert body['messages'][-1]['role'] == 'tool'
                assert body['messages'][-1]['tool_call_id'] == 'call_1'
            msg = {'role':'assistant','content':'Status read.'}
        return httpx.Response(200,json={'model':'test-model','choices':[{'message':msg,'finish_reason':'stop'}]})
    client = httpx.AsyncClient
    monkeypatch.setattr(model_agent.httpx,'AsyncClient',lambda **kwargs:client(transport=httpx.MockTransport(handler),**kwargs))
    config = model_agent.ModelConfig(endpoint=endpoint, model='test-model')
    async def invoke(name, values):
        invoked.append((name,values))
        return {'connected':True}
    async def run():
        checked = await model_agent.check_model(config)
        assert checked.get('status') != 'error'
        answer = await model_agent.run_turn(config,[{'role':'user','content':'Read state'}],
            [{'name':'robot_status','description':'Read state','inputSchema':{'type':'object','properties':{},'additionalProperties':False}}],
            invoke,lambda text:None)
        assert answer == 'Status read.'
    asyncio.run(run())
    assert invoked == [('robot_status',{})]
    assert len(requests) == 3
