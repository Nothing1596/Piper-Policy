"""Measurement-based relation checks and episode scoring for the piper sim (numpy only)."""
import numpy as np

_LATERAL_TOL = 0.01     # m, dataset-frame left/right/front/behind clearance
_SEAT_TOL = 0.012       # m, top placement seating tolerance
_FLOOR_TOL = 0.02       # m, lower containment tolerance, not forced floor seating
_SPEED_TOL = 0.03       # m/s, released-and-at-rest threshold
_LIFT_MIN = 0.03        # m, required measured transfer lift
_OPEN_Q, _CLOSE_Q = 0.14, 0.01
_RELATIONS = ("left", "right", "front", "behind", "top", "into")
_LATERAL = {"left": (0, -1.0), "right": (0, 1.0), "front": (1, -1.0), "behind": (1, 1.0)}

def _vec3(value, what):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)): raise ValueError("malformed %s" % what)
    return arr

def _num(value, what):
    if isinstance(value, bool): raise ValueError("malformed %s" % what)
    arr = np.asarray(value, dtype=float)
    if arr.shape != () or not np.isfinite(arr): raise ValueError("malformed %s" % what)
    return float(arr)

def _check(state):
    """Validate a state end to end; return (objects, dataset rotation)."""
    objs = state.get("objects") if isinstance(state, dict) else None
    if not isinstance(objs, dict) or not objs: raise ValueError("malformed state: missing objects")
    for name, obj in objs.items():
        if not isinstance(name, str) or not isinstance(obj, dict): raise ValueError("malformed object entry")
        aabb = obj.get("aabb")
        if not isinstance(aabb, dict): raise ValueError("missing aabb[%s]" % name)
        _vec3(obj.get("position"), "position[%s]" % name)
        lo = _vec3(aabb.get("lower"), "aabb.lower[%s]" % name)
        hi = _vec3(aabb.get("upper"), "aabb.upper[%s]" % name)
        if np.any(hi < lo) or _num(obj.get("speed_m_s"), "speed_m_s[%s]" % name) < 0.0:
            raise ValueError("inverted aabb or negative speed[%s]" % name)
        bounds = obj.get("container_bounds")
        if bounds is not None and not isinstance(bounds, dict): raise ValueError("bad container_bounds[%s]" % name)
        if bounds is not None:
            _vec3(bounds.get("lower"), "container_bounds.lower[%s]" % name)
            _vec3(bounds.get("upper"), "container_bounds.upper[%s]" % name)
        if obj.get("kind") == "drawer": _num(obj.get("qpos_m"), "qpos_m[%s]" % name)
    contacts = state.get("contacts")
    contacts = [] if contacts is None else contacts
    if not isinstance(contacts, (list, tuple)): raise ValueError("malformed contacts")
    for contact in contacts:
        names = (contact.get("object1"), contact.get("object2")) if isinstance(contact, dict) else ()
        if len(names) != 2 or not all(isinstance(n, str) for n in names): raise ValueError("malformed contact entry")
        if _num(contact.get("normal_force_n"), "normal_force_n") < 0.0: raise ValueError("negative normal_force_n")
    base = np.asarray(state.get("base_from_dataset"), dtype=float)
    if base.shape != (4, 4) or not np.all(np.isfinite(base)): raise ValueError("malformed base_from_dataset")
    if (not np.allclose(base[3], [0, 0, 0, 1]) or
        not np.allclose(base[:3, :3].T @ base[:3, :3], np.eye(3), atol=1e-6) or
        not np.isclose(np.linalg.det(base[:3, :3]), 1., atol=1e-6)):
        raise ValueError("base_from_dataset must be a rigid transform")
    return objs, base[:3, :3]

def _force(state, first, second):
    forces = [float(c["normal_force_n"]) for c in (state.get("contacts") or [])
              if {c["object1"], c["object2"]} == {first, second}]
    return max(forces) if forces else 0.0


def _contained(obj,ref):
    """Metric containment; contents may rest on other contained objects."""
    bounds=ref.get('container_bounds')
    if bounds is None:raise ValueError('Hollow container interior is not registered')
    floor,opening=np.asarray(bounds['lower'],float),np.asarray(bounds['upper'],float)
    lo,hi=np.asarray(obj['aabb']['lower'],float),np.asarray(obj['aabb']['upper'],float)
    inside=bool(np.all(lo[:2]>=floor[:2]-_LATERAL_TOL) and np.all(hi[:2]<=opening[:2]+_LATERAL_TOL))
    if ref.get('inner_circle'):
        circle=ref['inner_circle'];center=np.asarray(circle['center_xy_m'],float)
        radius=_num(circle['radius_m'],'inner radius')
        points=np.asarray(obj.get('grasp_frame',{}).get('footprint_xy_m'),float)
        if center.shape!=(2,) or radius<=0 or points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.all(np.isfinite(points)):
            raise ValueError('Measured collision footprint required for circular interior')
        inside=bool(np.all(np.linalg.norm(points-center,axis=1)<=radius+_LATERAL_TOL))
    return bool(inside and floor[2]<opening[2] and lo[2]>=floor[2]-_FLOOR_TOL
                and (lo[2]+hi[2])/2<opening[2])


def _container_support(state,source,reference):
    objs=state['objects'];ref=objs[reference]
    allowed={reference}
    for name,obj in objs.items():
        if name!=reference and obj.get('kind')=='dynamic' and obj['speed_m_s']<=_SPEED_TOL and _contained(obj,ref):
            allowed.add(name)
    reached={source};queue=[source]
    while queue:
        first=queue.pop()
        for second in allowed-reached:
            if _force(state,first,second)>0:
                if second==reference:return True
                reached.add(second);queue.append(second)
    return False

def _canon(action):
    if not isinstance(action, (list, tuple)) or not action or not isinstance(action[0], str):
        raise ValueError("malformed action %r" % (action,))
    if action[0] in ("open", "close"):
        if len(action) != 1: raise ValueError("%s takes no arguments" % action[0])
        return (action[0],)
    if action[0] != "pick_place": raise ValueError("unsupported action %r" % (action[0],))
    if len(action) != 4 or not all(isinstance(x, str) for x in action[1:]) or action[3] not in _RELATIONS:
        raise ValueError("malformed or unsupported pick_place action")
    return ("pick_place", action[1], action[2], action[3])

def relation_status(state, source, reference, relation):
    """Measured truth of relation(source, reference) in `state`; never inferred."""
    if relation not in _RELATIONS: raise ValueError("unsupported relation %r" % (relation,))
    if not isinstance(source, str) or not isinstance(reference, str) or source == reference:
        raise ValueError("source and reference must be distinct object names")
    objs, rot = _check(state)
    if source not in objs or reference not in objs: raise ValueError("unknown object in relation request")
    src, ref = objs[source], objs[reference]
    ps, pr = np.asarray(src["position"], float), np.asarray(ref["position"], float)
    ls, hs = np.asarray(src["aabb"]["lower"], float), np.asarray(src["aabb"]["upper"], float)
    lr, hr = np.asarray(ref["aabb"]["lower"], float), np.asarray(ref["aabb"]["upper"], float)
    if float(src["speed_m_s"]) > _SPEED_TOL: return False
    if relation in _LATERAL:
        axis, sign = _LATERAL[relation]
        delta = rot.T @ ((ls + hs - lr - hr) / 2.)
        return bool(sign * delta[axis] > _LATERAL_TOL and abs(delta[axis]) >= abs(delta[1-axis]))
    if relation == "top":
        if min(hs[0], hr[0]) <= max(ls[0], lr[0]) or min(hs[1], hr[1]) <= max(ls[1], lr[1]): return False
        return bool(abs(ls[2] - hr[2]) <= _SEAT_TOL and _force(state, source, reference) > 0.0)
    if ref.get("kind") not in ("container", "drawer"):
        raise ValueError("relation 'into' requires a container or drawer reference")
    return bool(_contained(src,ref) and _container_support(state,source,reference))

def _eval_step(expected, step):
    """Evidence and errors for one realized step, judged only on its own state_after."""
    ev, errs = {"status": None, "action": None}, []
    if not isinstance(step, dict): return ev, ["realized step is not a dict"]
    ev["status"] = step.get("status")
    if step.get("status") != "executed":
        errs.append("status %r is not 'executed'" % (step.get("status"),))
    if expected is None: return ev, errs + ["unexpected extra action"]
    try:
        got = _canon(step.get("action"))
    except ValueError as exc:
        return ev, errs + [str(exc)]
    ev["action"] = list(got)
    if got != expected: return ev, errs + ["realized %r != expected %r" % (list(got), list(expected))]
    try:
        objs, _ = _check(step.get("state_after"))
    except ValueError as exc:
        return ev, errs + ["malformed state_after: %s" % exc]
    if expected[0] in ("open", "close"):
        drawers = {n: o for n, o in objs.items() if o.get("kind") == "drawer"}
        if not drawers: return ev, errs + ["no drawer in state_after"]
        ev["drawers"] = {n: {"qpos_m": float(o["qpos_m"]), "travel_m": o.get("travel_m")} for n, o in drawers.items()}
        is_open = expected[0] == "open"
        ok = (any(q["qpos_m"] >= _OPEN_Q for q in ev["drawers"].values()) if is_open
              else any(q["qpos_m"] <= _CLOSE_Q for q in ev["drawers"].values()))
        if not ok: errs.append("%s requires qpos_m %s" % (expected[0], ">= 0.14" if is_open else "<= 0.01"))
        return ev, errs
    source, reference, relation = expected[1:]
    # Recompute from private measured snapshots. Executor booleans alone are
    # not evidence, and post-release z cannot measure a lift that already ended.
    try:
        initial,_=_check(step.get('state_before'))
        held,_=_check(step.get('state_held'))
        lifted,_=_check(step.get('state_lifted'))
        lift=float(lifted[source]['position'][2]-initial[source]['position'][2])
        bilateral=(source in step['state_held'].get('bilateral_contact_objects',[]) and
                   source in step['state_lifted'].get('bilateral_contact_objects',[]))
        released=source not in step['state_after'].get('bilateral_contact_objects',[])
        ev['grasp']={'bilateral_contact':bilateral,'lift_height_m':lift,'released':released}
        if not bilateral:errs.append('measured bilateral_contact is not True')
        if lift<_LIFT_MIN:errs.append('measured lift_height_m < 0.03')
        if not released:errs.append('measured released is not True')
    except (ValueError,KeyError,TypeError) as exc:
        ev['grasp']=None;errs.append('missing or malformed measured grasp snapshots: '+str(exc))
    try:
        holds = relation_status(step["state_after"], source, reference, relation)
        ev["relation"] = {"relation": relation, "source": source, "reference": reference, "holds": bool(holds)}
        if not holds: errs.append("%s(%s, %s) does not hold in state_after" % (relation, source, reference))
        ev["source_speed_m_s"] = speed = float(objs[source]["speed_m_s"])
        if relation in _LATERAL and speed > _SPEED_TOL: errs.append("released source speed %g > 0.03" % speed)
    except ValueError as exc:
        ev["relation"] = {"relation": relation, "source": source, "reference": reference, "holds": None}
        errs.append(str(exc))
    return ev, errs

def score_episode(reference_actions, realized_steps, before, after):
    """Score realized steps against canonical actions; each step keeps its own state_after."""
    if not isinstance(reference_actions, (list, tuple)) or not isinstance(realized_steps, (list, tuple)):
        raise ValueError("reference_actions and realized_steps must be sequences")
    expected = [_canon(a) for a in reference_actions]
    _check(before)
    objs_after, _ = _check(after)
    steps = []
    for index, step in enumerate(realized_steps):
        want = expected[index] if index < len(expected) else None
        ev, errs = _eval_step(want, step)
        steps.append({"index": index, "expected": None if want is None else list(want),
                      "realized": step.get("action") if isinstance(step, dict) else None,
                      "success": want is not None and not errs, "evidence": ev, "errors": errs})
    steps += [{"index": i, "expected": list(expected[i]), "realized": None, "success": False, "evidence": {},
               "errors": ["missing expected action"]} for i in range(len(realized_steps), len(expected))]
    sequence_success = len(steps) == len(expected) and all(
        s["expected"] is not None and s["evidence"].get("action") == s["expected"] for s in steps)
    final = {"relations_before": {}, "relations": {}, "speed_m_s": {}, "errors": []}
    for want in [w for w in expected if w[0] == "pick_place"]:
        key = "%s %s %s" % (want[3], want[1], want[2])
        for label, state in (("relations_before", before), ("relations", after)):
            try: final[label][key] = bool(relation_status(state, want[1], want[2], want[3]))
            except ValueError as exc:
                final[label][key] = None
                final["errors"].append("%s %s: %s" % (label, key, exc))
        obj = objs_after.get(want[1])
        final["speed_m_s"][want[1]] = float(obj["speed_m_s"]) if isinstance(obj, dict) else None
    final["drawers"] = {n: {"qpos_m": o.get("qpos_m"), "travel_m": o.get("travel_m")} for n, o in objs_after.items() if o.get("kind") == "drawer"}
    return {"sequence_success": bool(sequence_success), "steps": steps, "final_state": final,
            "success": bool(sequence_success and all(s["success"] for s in steps))}
