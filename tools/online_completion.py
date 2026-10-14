"""Experiment-only terminal evidence. No motion, GT feedback, or new robot API."""
import math
import re


def distance(a,b):
    if len(a)!=3 or len(b)!=3 or not all(math.isfinite(x) for x in [*a,*b]):
        raise ValueError('invalid position')
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def retreat_evidence(chain):
    """Opening alone is not retreat. Use actual samples, not commanded targets."""
    try:
        opens=[i for i,r in enumerate(chain) if r['tool'] in ('set_gripper','robot_gripper')
               and r['actual_result']['status']=='succeeded'
               and r['state_after']['robot']['gripper_width_m']>=.06
               and r['state_after']['robot']['gripper_width_m']>r['state_before']['robot']['gripper_width_m']+.001]
        if not opens:return {'ok':False,'reason':'No measured release followed by retreat'}
        index=opens[-1];later=chain[index+1:]
        moved=any(r['tool'] in ('move_to','move_by','move_linear','rotate','robot_move_joints')
                  and r['actual_result']['status']=='succeeded' for r in later)
        release=chain[index]['state_after']['robot'];current=chain[-1]['state_after']['robot']
        delta=distance(current['tcp_pos'],release['tcp_pos'])
        return {'ok':bool(moved and delta>=.03 and current['gripper_width_m']>=.06),
                'retreat_distance_m':delta,'minimum_m':.03,'release_chain_index':index,
                'reason':'Measured displacement after release; not a collision-free path guarantee'}
    except (KeyError,TypeError,ValueError,IndexError):
        return {'ok':False,'reason':'Missing/invalid measured retreat evidence'}


def stable_after_retreat(samples,chain,sources):
    """Known Piper scene robot geom names; final snapshots remain private."""
    result={'ok':False,'retreat':retreat_evidence(chain),'reasons':[],
            'thresholds':{'span_s':2.,'max_displacement_m':.002,'max_speed_m_s':.005,'max_rotation_deg':2.}}
    try:
        if len(samples)<3:raise ValueError('Need at least three post-loop samples')
        times=[s['sim_time_s'] for s in samples]
        if not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])) or times[-1]-times[0]<2:
            raise ValueError('Need advancing simulation samples spanning at least 2 seconds')
        if len({s['epoch'] for s in samples})!=1:raise ValueError('Simulator epoch changed')
        if not result['retreat']['ok']:result['reasons'].append('retreat_not_verified')
        result['sample_span_s']=times[-1]-times[0];result['objects']={}
        for source in sources:
            obs=[s['objects'][source] for s in samples]
            if any(o['kind']!='dynamic' for o in obs):raise ValueError('Expected dynamic source')
            drift=max(distance(a['position'],b['position']) for a in obs for b in obs)
            speeds=[o['speed_m_s'] for o in obs]
            if not all(math.isfinite(v) and v>=0 for v in speeds):raise ValueError('Invalid speed')
            angle=0.
            for a in obs:
                for b in obs:
                    qa,qb=a['quaternion'],b['quaternion']
                    if len(qa)!=4 or len(qb)!=4 or not all(math.isfinite(x) for x in [*qa,*qb]):raise ValueError('Invalid rotation')
                    norm=math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
                    if norm<1e-9:raise ValueError('Invalid rotation norm')
                    angle=max(angle,math.degrees(2*math.acos(min(1.,abs(sum(x*y for x,y in zip(qa,qb)))/norm))))
            robot_contacts=[]
            for i,s in enumerate(samples):
                if source in s['bilateral_contact_objects']:robot_contacts.append({'sample':i,'bilateral':True})
                for c in s['contacts']:
                    a,b=c['object1'],c['object2'];other=b if a==source else a if b==source else ''
                    if re.match(r'^(gripper_|link[1-6]_|base_link_|flange_)',other):
                        robot_contacts.append({'sample':i,'geometry':other,'normal_force_n':c['normal_force_n']})
            ok=not robot_contacts and drift<=.002 and max(speeds)<=.005 and angle<=2.
            result['objects'][source]={'ok':ok,'max_displacement_m':drift,'max_speed_m_s':max(speeds),
                                       'max_rotation_deg':angle,'robot_contacts':robot_contacts}
            if not ok:result['reasons'].append('contact_or_unstable:'+source)
        result['ok']=bool(sources and not result['reasons'])
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        result['reasons'].append('invalid_evidence:'+str(exc))
    return result


def tabletop_return(samples, transfer, *, tolerance_m=.005):
    """Direct tabletop return, separate from stability or stacked-task scoring.

    Compare against the source's measured, table-supported starting bottom.
    No GT goal pose or new runtime observation is supplied to the model.
    """
    result={'ok':False,'reason':None,'bottom_tolerance_m':tolerance_m}
    try:
        if not math.isfinite(tolerance_m) or tolerance_m<=0:raise ValueError('Invalid tolerance')
        if len(samples)<3:raise ValueError('Missing final samples')
        source=transfer['source'];before=transfer['state_before']
        initial_bottom=before['objects'][source]['aabb']['lower'][2]
        if not math.isfinite(initial_bottom):raise ValueError('Invalid initial bottom')
        def table_force(state):
            values=[c['normal_force_n'] for c in state['contacts']
                    if {c['object1'],c['object2']}=={source,'table_top'}]
            if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid contact force')
            return sum(values)
        if table_force(before)<=0:raise ValueError('Initial direct tabletop support unverified')
        forces=[];offsets=[]
        for state in samples:
            force=table_force(state)
            bottom=state['objects'][source]['aabb']['lower'][2]
            if not math.isfinite(bottom):raise ValueError('Invalid final bottom')
            offsets.append(abs(bottom-initial_bottom));forces.append(force)
        result.update(table_normal_forces_n=forces,bottom_offsets_m=offsets)
        if min(forces)<=0:raise ValueError('Final direct tabletop support unverified')
        if max(offsets)>tolerance_m:raise ValueError('Final support height differs from starting tabletop')
        result['ok']=True
    except (KeyError,TypeError,ValueError,IndexError) as exc:result['reason']=str(exc)
    return result
