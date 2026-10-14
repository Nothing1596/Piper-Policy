"""Image-grounded operation proposals, with no execution authority."""
import json
from copy import deepcopy
from pathlib import Path


POINT = {"type":"array", "items":{"type":"number","minimum":0,"maximum":1},"minItems":2,"maxItems":2}
DECISION_SCHEMA = {"type":"object","properties":{
    "action":{"type":"string","enum":["pick_place","push","observe","query_demo","done","stop"]},
    "stage_id":{"type":"string"}, "object_point":POINT,"destination_point":POINT,
    "object_candidate_id":{"type":"string"},"destination_candidate_id":{"type":"string"},
    "object_label":{"type":"string"},"expected_effect":{"type":"string"},
    "reason":{"type":"string"},"evidence_refs":{"type":"array","items":{"type":"string"}},
    "query":{"type":"string"}},"required":["action","stage_id","object_point","destination_point",
    "object_label","object_candidate_id","destination_candidate_id","expected_effect","reason","evidence_refs","query"],"additionalProperties":False}


class VisionPlanner:
    def __init__(self, model):
        self.model = model

    def decide(self, *, task, observation_path, robot_state, demo=None, history=(), candidates=()):
        frames = (demo or {}).get("frames", [])
        # Demonstration compiler has already established the operation sequence.
        # Include up to four referenced visual anchors and one current live view.
        if len(frames)>4:
            indices = sorted({0, len(frames)//3, 2*len(frames)//3, len(frames)-1})
            frames = [frames[i] for i in indices]
        queries=[h['demo_answer'] for h in history if h.get('demo_answer')]
        if queries:
            answer=queries[-1]
            recalled=[f for f in answer['frames'] if f['frame_id'] in answer['evidence_refs']][:3]
            frames=frames[:2]+recalled
        images = [Path(f["image_path"]) for f in frames] + [Path(observation_path)]
        allowed = {"live"}|{f["frame_id"] for f in frames}
        schema = deepcopy(DECISION_SCHEMA)
        schema['properties']['evidence_refs']['items']['enum'] = sorted(allowed)
        schema['properties']['evidence_refs']['minItems'] = 1
        placement_regions=[c['id'] for c in candidates if c.get('color')=='green']
        schema['allOf']=[{'if':{'properties':{'action':{'const':'pick_place'}}},
            'then':{'properties':{'destination_candidate_id':{'enum':placement_regions}}} if placement_regions else
                   {'not':{'properties':{'action':{'const':'pick_place'}}}}}]
        historical = {k:v for k,v in (demo or {}).items() if k not in ("frames", "source_path")}
        historical['resolved_queries']=[{k:v for k,v in q.items() if k!='frames'} for q in queries]
        outcomes=[]
        for entry in list(history)[-8:]:
            decision=entry.get('decision',{})
            outcomes.append({'action':decision.get('action'),'stage_id':decision.get('stage_id'),
                             'object':decision.get('object_label'),'expected_effect':decision.get('expected_effect'),
                             'execution_status':entry.get('execution',{}).get('status'),
                             'independent_verification':entry.get('verification'),
                             'demo_answer':entry.get('demo_answer'),'outcome':entry.get('outcome')})
        prompt = (
            "You propose one bounded robot operation at a time in a tabletop task. "
            "Historical human demonstration is reference evidence, not the live scene. "
            "Use its task-specific sequence and constraints; adapt to current object locations. "
            "Images before the final image are historical demonstration references; the FINAL image is LIVE. "
            "Point coordinates are [u/image_width, v/image_height] in the LIVE image. "
            "Select object_candidate_id and destination_candidate_id from live_measured_candidates whenever the region exists. "
            "These IDs identify measured regions, not semantic classes. Use empty ID only for a free-space destination or no motion. "
            "Pick object center on visible top surface; place/push destination is a visible free table/tray point. "
            "Allowed skills: pick_place for a rigid small block; push for a horizontal table displacement. "
            "In this first-version colored-block profile, pick_place requires a measured green destination region ID. "
            "Do not substitute a free-space point for a named region; if no destination region is observed, choose observe or stop. "
            "For push, prefer a measured free_table_destination with matching source_candidate_id and requested image_direction; these targets are 8 cm away. "
            "Use observe for uncertainty, query_demo for missing procedural evidence, stop for unsupported tasks. "
            "Do not claim success from commanded motion; done is a hypothesis requiring independent verification. "
            "Read execution_history carefully: independent_verification.verdict=supported confirms the stated effect. "
            "Do not repeat that completed transfer. When all objects requested by the task have supported effects, choose done. "
            "For observe/query_demo/done/stop use [0,0] points. stage_id references historical stage if available. "
            "Use evidence_refs=['live'] plus frame IDs from allowed_current_evidence_refs only. "
            "Historical stage citations describe archived evidence; a cited frame may not be attached to this request. "
            "Do not cite an unattached archived frame as current visual evidence. Use query_demo if its pixels are needed.\n"
            + json.dumps({"task":task,"historical":historical,"historical_frames":[
                {"frame_id":f["frame_id"],"timestamp_s":f["timestamp_s"]} for f in frames],
                "allowed_current_evidence_refs":sorted(allowed),
                "live_measured_candidates":list(candidates),
                "robot_state":{"tcp":robot_state.get("tcp"),"ready":robot_state.get("ready"),
                    "gripper_width_m":robot_state.get("robot",{}).get("gripper_width_m")},
                "execution_history":outcomes}, ensure_ascii=False))
        if not demo:
            prompt += '\nNO DEMONSTRATION WAS PROVIDED. Do not invent historical stages or references. evidence_refs must be ["live"]. stage_id can be an empty string.'
        result = self.model.infer(prompt, images, schema)
        if not set(result["evidence_refs"]) <= allowed or "live" not in result["evidence_refs"]:
            result = self.model.infer(prompt + '\nCORRECTION: evidence_refs must include live and may contain only these exact IDs: ' + json.dumps(sorted(allowed)), images, schema)
            if not set(result["evidence_refs"]) <= allowed or "live" not in result["evidence_refs"]:
                raise ValueError("invalid_decision_evidence")
        stages = {s["id"] for s in (demo or {}).get("stages",[])}
        if result["action"] in ("pick_place","push") and stages and result["stage_id"] not in stages:
            raise ValueError("unknown_demo_stage")
        ids={c['id'] for c in candidates}
        if result['action'] in ('pick_place','push'):
            if result['action']=='pick_place' and result['destination_candidate_id'] not in placement_regions:
                raise ValueError('pick_place_requires_measured_destination_region')
            if result['object_candidate_id'] not in ids or (result['destination_candidate_id'] and result['destination_candidate_id'] not in ids):
                raise ValueError('unknown_live_candidate')
        return result
