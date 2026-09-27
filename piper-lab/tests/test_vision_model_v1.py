import json
from pathlib import Path
import httpx
import pytest
from PIL import Image
from piperlab.models.lmstudio import LocalVisionModel, ModelError


def test_image_budget_and_remote_endpoint_rejected(tmp_path):
    with pytest.raises(ValueError):
        LocalVisionModel(base_url="https://example.com/v1")
    with pytest.raises(ModelError, match="image_budget"):
        LocalVisionModel(max_images=1).infer("x",[Path("x"),Path("y")],{"type":"object"})


def test_native_exact_fence_validation_and_truncation(monkeypatch,tmp_path):
    import piperlab.models.lmstudio as module
    original=httpx.Client
    path=tmp_path/'frame.png';Image.new('RGB',(32,32),'red').save(path)
    seen=[]
    def serve(request):
        body=json.loads(request.content);seen.append(body)
        return httpx.Response(200,json={"model_instance_id":"vision",
            "output":[{"type":"message","content":'```json\n{"value": 1}\n```'}],
            "stats":{"total_output_tokens":10}})
    monkeypatch.setattr(module.httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(serve),**kw))
    schema={"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":False}
    model=LocalVisionModel(log_dir=tmp_path/'calls')
    assert model.infer('inspect',[path],schema)=={"value":1}
    assert seen[0]['reasoning']=='off'
    assert any(i['type']=='image' for i in seen[0]['input'])
    assert len(list((tmp_path/'calls').glob('*.json')))==1
    with pytest.raises(ModelError,match='incomplete'):
        LocalVisionModel(max_output_tokens=10).infer('inspect',[path],schema)


def test_artifact_identity_rejects_changed_bytes(monkeypatch,tmp_path):
    import hashlib
    import piperlab.models.lmstudio as module
    original=httpx.Client
    def serve(request):
        return httpx.Response(200,json={'models':[{'key':'test/model','size_bytes':4,
            'capabilities':{'vision':True},'loaded_instances':[{'id':'test','config':{}}]}]})
    monkeypatch.setattr(module.httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(serve),**kw))
    weight=tmp_path/'weights.gguf';weight.write_bytes(b'abcd')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'model_key':'test/model','files':[{'path':str(weight),'bytes':4,
        'sha256':hashlib.sha256(b'abcd').hexdigest()}]}))
    assert LocalVisionModel(model='test').discover_identity()['cache_identity_complete'] is False
    assert LocalVisionModel(model='test',artifact_manifest=manifest).discover_identity()['cache_identity_complete']
    weight.write_bytes(b'abce')
    with pytest.raises(ModelError,match='hash_mismatch'):
        LocalVisionModel(model='test',artifact_manifest=manifest).discover_identity()
