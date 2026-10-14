import json
from pathlib import Path
import httpx
import pytest
from PIL import Image
from piperlab.models import providers as p

SCHEMA = {'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}


def response(provider, text='{"answer":"visible"}'):
    return {
        'openai': {'status':'completed','output':[{'type':'message','status':'completed','content':[{'type':'output_text','text':text}]}]},
        'openai-compatible': {'choices':[{'finish_reason':'stop','message':{'content':text}}]},
        'anthropic': {'stop_reason':'end_turn','content':[{'type':'text','text':text}]},
        'gemini': {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':text}]}}]},
    }[provider]


def mock_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(p.httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.parametrize('provider,endpoint,header', [
    ('openai','/v1/responses','authorization'), ('openai-compatible','/v1/chat/completions','authorization'),
    ('anthropic','/v1/messages','x-api-key'), ('gemini','/v1/models/test-model:generateContent','x-goog-api-key')])
def test_multimodal_wire_contract(monkeypatch, tmp_path, provider, endpoint, header):
    monkeypatch.setenv('TEST_API_KEY', 'test-secret')
    image = tmp_path/'f000001.png'; Image.new('RGB',(8,8)).save(image)
    requests = []
    def handle(req):
        requests.append(req)
        assert req.url.path == endpoint and not req.url.query
        assert req.headers[header].endswith('test-secret')
        body = json.loads(req.content)
        assert body['model'] == 'test-model' if provider != 'gemini' else 'model' not in body
        assert 'iVBOR' in req.content.decode()  # actual PNG pixels, not just a path
        assert 'f000001' in req.content.decode()
        return httpx.Response(200,json=response(provider))
    mock_transport(monkeypatch,handle)
    model = p.ApiVisionModel(p.ModelProfile(provider=provider,model='test-model',base_url='https://example.invalid/v1',api_key_env='TEST_API_KEY'),log_dir=tmp_path/'calls')
    assert model.discover_identity()['cache_identity_complete'] is False
    assert not requests
    assert model.infer('Describe', [image], SCHEMA) == {'answer':'visible'}
    assert len(requests) == 1 and model.last_call['status'] == 'ok'
    assert 'test-secret' not in next((tmp_path/'calls').glob('*.json')).read_text()


@pytest.mark.parametrize('provider', ['openai','openai-compatible','anthropic','gemini'])
def test_reject_truncated_json_even_if_parseable(monkeypatch, provider):
    raw=response(provider)
    if provider=='openai': raw['status']='incomplete'
    elif provider=='openai-compatible': raw['choices'][0]['finish_reason']='length'
    elif provider=='anthropic': raw['stop_reason']='max_tokens'
    else: raw['candidates'][0]['finishReason']='MAX_TOKENS'
    mock_transport(monkeypatch,lambda req:httpx.Response(200,json=raw))
    model=p.ApiVisionModel(p.ModelProfile(provider=provider,model='test',base_url='http://127.0.0.1/v1'))
    with pytest.raises(p.ModelError): model.infer('x',[],SCHEMA)
    assert model.last_call['status']=='error'


@pytest.mark.parametrize('body', ['{}','{"answer":2}','{"answer":NaN}','not json'])
def test_response_schema_rejection(monkeypatch,body):
    mock_transport(monkeypatch,lambda req:httpx.Response(200,json=response('openai',body)))
    model=p.ApiVisionModel(p.ModelProfile(provider='openai',model='test',base_url='http://localhost/v1'))
    with pytest.raises(p.ModelError):model.infer('x',[],SCHEMA)


@pytest.mark.parametrize('provider', ['openai','openai-compatible','anthropic','gemini'])
def test_refusals_and_unexpected_tools(monkeypatch,provider):
    raw=response(provider)
    if provider=='openai': raw['output'][0]['content']=[{'type':'refusal','refusal':'cannot answer'}]
    elif provider=='openai-compatible': raw['choices'][0]['message']['tool_calls']=[{'id':'unrequested'}]
    elif provider=='anthropic': raw['content'].append({'type':'tool_use','name':'unrequested'})
    else: raw['candidates'][0]['content']['parts'].append({'functionCall':{'name':'unrequested'}})
    mock_transport(monkeypatch,lambda req:httpx.Response(200,json=raw))
    model=p.ApiVisionModel(p.ModelProfile(provider=provider,model='test',base_url='http://localhost/v1'))
    with pytest.raises(p.ModelError):model.infer('x',[],SCHEMA)


def test_no_redirect_retry_or_error_body_leak(monkeypatch,tmp_path):
    secret='secret"with\\escapes'
    monkeypatch.setenv('TEST_API_KEY',secret)
    calls=[]
    def handler(req):
        calls.append(req)
        return httpx.Response(307,headers={'Location':'https://other.invalid'},text=secret)
    mock_transport(monkeypatch,handler)
    model=p.ApiVisionModel(p.ModelProfile(provider='openai',model='test',api_key_env='TEST_API_KEY'),log_dir=tmp_path)
    with pytest.raises(p.ModelError,match='provider_http_307'): model.infer(secret,[],SCHEMA)
    assert len(calls)==1 and model.last_call['prompt']=='[REDACTED]'
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['prompt']=='[REDACTED]'


@pytest.mark.parametrize('kwargs', [
    {'provider':'other'}, {'provider':'openai'}, {'provider':'lmstudio','base_url':'https://remote.invalid'},
    {'provider':'openai','model':'x','base_url':'http://remote.invalid'},
    {'provider':'openai','model':'x','base_url':'https://user:pass@remote.invalid'},
    {'provider':'openai','model':'x','max_images':True}, {'timeout_s':float('nan')},
    {'provider':'gemini','model':'x','structured_output':'json_object'}])
def test_invalid_profiles(kwargs):
    with pytest.raises(ValueError):p.ModelProfile(**kwargs)


def test_local_default_and_explicit_json_fallback(tmp_path):
    assert isinstance(p.create_model(p.ModelProfile()),p.LocalVisionModel)
    profile=p.ModelProfile(provider='openai-compatible',model='vision',base_url='http://localhost:8000/v1',structured_output='json_object')
    assert p.ApiVisionModel(profile)._payload('x',[],SCHEMA)[1]['response_format']=={'type':'json_object'}
    file=tmp_path/'bad.json';file.write_text('{"api_key":"do-not-inline"}')
    with pytest.raises(TypeError):p.ModelProfile.load(file)


def test_missing_key_cancellation_and_image_budget(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    model=p.ApiVisionModel(p.ModelProfile(provider='openai',model='test'))
    with pytest.raises(p.ModelError,match='missing_api_key'): model.discover_identity()
    with pytest.raises(p.ModelError,match='image_budget'):model.infer('x',[Path('unused')]*7,SCHEMA)
    model.cancel()
    with pytest.raises(p.ModelError,match='cancelled'):model.infer('x',[],SCHEMA)


def test_real_compiler_schema_optional_fields_roundtrip():
    from piperlab.demonstration.compiler import SELECT_SCHEMA,SEMANTIC_SCHEMA
    import copy
    before=copy.deepcopy(SELECT_SCHEMA)
    strict=p.strict_schema(SELECT_SCHEMA)
    row={'frame_id':'f1','keep':True,'reason':'visible', 'operation':None,'stage_hint':None,
         'object_roles':None,'visible_outcome':None}
    value={'selections':[row]}
    p.Draft202012Validator(strict).validate(value)
    restored=p.restore_optional(value,SELECT_SCHEMA)
    assert restored=={'selections':[{'frame_id':'f1','keep':True,'reason':'visible'}]}
    p.Draft202012Validator(SELECT_SCHEMA).validate(restored)
    assert before==SELECT_SCHEMA
    assert p.strict_schema(SEMANTIC_SCHEMA)['additionalProperties'] is False


def test_compiler_through_api_adapter(monkeypatch,tmp_path):
    from test_video_v1 import _write_synthetic_video
    from piperlab.demonstration.compiler import compile_demo
    video=_write_synthetic_video(tmp_path/'synthetic.mp4',frames=12)
    calls=[]
    def handler(req):
        body=json.loads(req.content);schema=body['text']['format']['schema']
        calls.append(schema)
        if 'selections' in schema['properties']:
            ids=schema['properties']['selections']['items']['properties']['frame_id']['enum']
            value={'selections':[{'frame_id':i,'keep':True,'reason':'synthetic mock',
                                 'operation':None,'stage_hint':None,'object_roles':None,'visible_outcome':None} for i in ids]}
        else:
            ids=schema['properties']['stages']['items']['properties']['evidence_refs']['items']['enum']
            value={'stages':[{'id':'s1','operation':'observe','object_roles':['red block'],
                             'preconditions':[],'expected_effects':[],'evidence_refs':ids,'uncertainty':'mock protocol only'}],
                   'goal_constraints':[],'unknowns':['No real inference performed'],'outcome_verdict':'unknown','summary':'synthetic protocol integration'}
        p.Draft202012Validator(schema).validate(value)
        return httpx.Response(200,json=response('openai',json.dumps(value)))
    mock_transport(monkeypatch,handler)
    model=p.ApiVisionModel(p.ModelProfile(provider='openai',model='mock-vision',base_url='http://localhost/v1'),log_dir=tmp_path/'calls')
    result=compile_demo(str(video),str(tmp_path/'bundle'),'Observe synthetic video',model)
    assert len(calls)>=2 and result['outcome_verdict']=='unknown'
    assert result['stages'] and (tmp_path/'bundle/demo.json').is_file()
