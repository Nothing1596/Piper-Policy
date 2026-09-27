import copy
import json
from piperlab.demonstration.compiler import _detection_brief


def test_large_diagnostics_do_not_enter_prompt_or_mutate_evidence():
    source={'count':30,'regions':[{'label':'cup','score':i/30,'bbox_xyxy':[1.11111,2,3,4],
        'track_id':i,'track_epoch':2} for i in range(30)],
        'detector_report':{'dropped':[{'index':i,'reason':'nms_overlap'} for i in range(10000)]}}
    saved=copy.deepcopy(source)
    brief=_detection_brief(source)
    assert source==saved
    assert len(brief['regions'])==12 and brief['omitted_regions']==18
    assert brief['regions'][0]['track_id']==29
    assert 'fallible' in brief['interpretation']
    assert 'detector_report' not in brief
    assert len(json.dumps(brief))<2500


def test_absent_detector_is_not_fabricated():
    assert _detection_brief(None) is None
    assert _detection_brief({'count':0,'regions':[]})['regions']==[]
