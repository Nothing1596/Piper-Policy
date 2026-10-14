"""Demonstration-conditioned visual loop with recordings and explicit incomplete outcomes."""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from PIL import Image
from .grounding import ground_pixel,scene_unchanged
from .planner import VisionPlanner
from .skills import SkillRunner
from .verify import verify_block_destination,recheck_latest_effects
from .scene import measured_candidates,resolve_object_point,free_region_pixels,footprint_is_free
from .observation import ObservationStream
from .demo_query import query_demo
from .context import active_demo,segment_count
from ..models.worker import ModelWorker,Ticket


def _save_frame(observation,path):
    Image.fromarray(observation['rgb']).save(path,quality=90)


def _identity(observation):
    return (observation.get('instance_id'),observation.get('connection_epoch'),observation['epoch'])


def run_policy(executor,model,*,task,output_dir,demo=None,max_decisions=20,skill_config=None):
    if not isinstance(max_decisions,int) or isinstance(max_decisions,bool) or max_decisions<1:
        raise ValueError('max_decisions_must_be_positive')
    output=Path(output_dir).resolve()
    started_at=time.time();started_monotonic=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    planner=VisionPlanner(model)
    skills=SkillRunner(executor,**(skill_config or {}))
    camera=ObservationStream(executor,output)
    worker=ModelWorker(model)
    query_round=0
    segment_index=0
    def monitor_camera():
        sample=camera.latest()
        if _identity(sample)!=execution_epoch:
            raise RuntimeError('camera_epoch_changed_during_skill')
    skills.visual_monitor=monitor_camera
    history=[]
    verified_stages=set()
    report={'task':task,'demo_id':(demo or {}).get('demo_id'),'model_identity':model.identity,
            'status':'incomplete','decisions':0,'history':history,'provenance':'visual_closed_loop_simulation',
            'started_at':started_at,'max_decisions':max_decisions}
    def record(value):
        history.append(value)
        with (output/'steps.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    try:
        capabilities=executor.request('GET','/v1/capabilities')
        if capabilities.get('backend')!='mujoco':
            raise ValueError('first_version_requires_physical_simulation')
        if demo and demo.get('outcome_verdict')=='unresolved':
            raise ValueError('unresolved_demonstration_requires_review')
        camera.start()
        for step in range(max_decisions):
            current_demo=active_demo(demo,segment_index)
            before=camera.latest()
            path=output/f'{step:03d}-before.jpg';_save_frame(before,path)
            state=executor.state()
            if state['active_job_id']:
                raise ValueError('executor_already_busy')
            ticket=Ticket(str(uuid.uuid4()),before['state_version'],before['epoch'],time.monotonic()+180)
            worker.submit_call(ticket,planner.decide,task=task,observation_path=path,robot_state=state,demo=current_demo,history=history,
                               candidates=before['candidates'])
            while True:
                latest=camera.latest()
                response=worker.poll(state_version=latest['state_version'],epoch=latest['epoch'])
                if response['status']=='expired_pending':
                    raise TimeoutError('planner_deadline')
                if response['status']!='pending':break
                time.sleep(.05)
            if response['status']=='discarded':
                record({'step':step,'outcome':'discarded_stale_model_response','request_id':ticket.request_id})
                continue
            if response['status']!='ok':raise RuntimeError(response.get('reason','planner_failed'))
            decision=response['value']
            report['decisions']+=1
            entry={'step':step,'decision':decision,'before_image':str(path),
                   'observation_stamp':before['source_stamp_s'],'epoch':before['epoch'],
                   'request_id':ticket.request_id,'state_version':ticket.state_version}
            action=decision['action']
            entry['segment_index']=segment_index
            if action in ('pick_place','push'):
                repeated=any(h.get('verification',{}).get('verdict')=='supported' and
                    all(h.get('decision',{}).get(k)==decision[k] for k in
                        ('action','object_candidate_id','destination_candidate_id','stage_id'))
                    for h in history)
                if repeated and decision['destination_candidate_id']:
                    entry['outcome']='rejected_repeated_verified_effect'
                    record(entry);report['status']='stopped_repeated_completed_action';break
            if action in ('stop','done'):
                if action=='done':
                    completion_frame=camera.latest()
                    completion_path=output/f'{step:03d}-completion.jpg'
                    _save_frame(completion_frame,completion_path)
                    entry['completion_recheck']=recheck_latest_effects(completion_frame,history)
                    entry['completion_recheck']['evidence_refs']=[str(completion_path)]
                    if entry['completion_recheck']['verdict']!='supported':
                        entry['outcome']='completion_not_supported_by_current_observation'
                        record(entry);report['status']='completion_recheck_failed';break
                if action=='done' and segment_index+1<segment_count(demo):
                    entry['outcome']='segment_done_pending_task_evaluation'
                    record(entry);segment_index+=1;continue
                needed={s['id'] for s in (demo or {}).get('stages',[]) if s.get('operation') in ('pick_place','push')}
                report['status']='model_declared_done_pending_task_evaluation' if action=='done' else 'stopped'
                entry['verified_stages']=sorted(verified_stages)
                entry['unverified_stages']=sorted(needed-verified_stages)
                record(entry);break
            if action=='observe':
                entry['outcome']='observation_requested';record(entry);continue
            if action=='query_demo':
                if query_round>=3 or demo is None:
                    entry['outcome']='unresolved_demo_query';record(entry)
                    report['status']='needs_demo_evidence';break
                answer=query_demo(current_demo,decision['query'],decision['stage_id'],model,output,query_round)
                query_round+=1
                entry['outcome']='demonstration_query_answered'
                entry['demo_answer']=answer
                record(entry);continue
            # Inference consumes time. Re-observe, compare scene and ground only from fresh depth.
            fresh=camera.latest()
            if _identity(fresh)!=_identity(before) or not scene_unchanged(before['rgb'],fresh['rgb']):
                entry['outcome']='discarded_scene_changed';record(entry);continue
            h,w=fresh['rgb'].shape[:2]
            points=[]
            regions={c['id']:c for c in measured_candidates(fresh)}
            for key in ('object_point','destination_point'):
                u,v=decision[key]
                pixel=[u*(w-1),v*(h-1)]
                candidate_id=decision[key.replace('_point','_candidate_id')]
                if candidate_id:
                    if candidate_id not in regions:
                        raise ValueError('candidate_disappeared')
                    region=regions[candidate_id]
                    pixel=region['pixel_xy']
                    entry[key.replace('_point','_region')]=region
                    if key=='destination_point' and action=='pick_place' and region['color']=='green':
                        from piperx_middleware.kinematics import PiperKinematics
                        cfg=capabilities['cartesian_primitives']
                        kin=PiperKinematics(cfg['tcp_offset_m'],cfg['tcp_offset_rpy_deg'])
                        chosen=None
                        for proposal in free_region_pixels(fresh['rgb'],region,include_grid=True):
                            try:
                                target=ground_pixel(proposal,fresh['depth'],fresh['camera_info'],
                                    source_stamp_s=fresh['source_stamp_s'],epoch=fresh['epoch'],max_age_s=1,now=fresh['source_stamp_s'])
                                if not footprint_is_free(fresh['rgb'],region,target,fresh['camera_info'],half_size=skills.placement_half_size):continue
                                xyz=list(target.xyz_m);xyz[2]=max(points[0].xyz_m[2],xyz[2])+skills.hover_m
                                kin.solve(xyz,skills.rpy,state['robot']['q_deg'])
                                xyz[2]=target.xyz_m[2]+skills.fingertip_clearance_m+.002
                                kin.solve(xyz,skills.rpy,state['robot']['q_deg'])
                                chosen=proposal;break
                            except (ValueError,RuntimeError):continue
                            except Exception as exc:
                                if getattr(exc,'code',None)=='ik_no_solution':continue
                                raise
                        if chosen is None:raise ValueError('no_free_reachable_placement_in_selected_region')
                        pixel=chosen
                        entry['placement_selection']={'pixel_xy':pixel,'method':'metric_footprint_free_space_and_ik',
                                                      'region_id':candidate_id,'footprint_half_size_m':skills.placement_half_size}
                elif key=='object_point':
                    region=resolve_object_point(fresh['rgb'],decision['object_label'],pixel)
                    pixel=region['pixel_xy']
                    entry['object_region']=region
                points.append(ground_pixel(pixel,fresh['depth'],fresh['camera_info'],
                    source_stamp_s=fresh['source_stamp_s'],epoch=fresh['epoch'],max_age_s=1,now=fresh['source_stamp_s']))
            entry['grounding']=[p.__dict__ for p in points]
            try:
                skills.prepare(action,*points)
                confirmed=camera.latest()
                if _identity(confirmed)!=_identity(fresh) or not scene_unchanged(fresh['rgb'],confirmed['rgb']):
                    entry['outcome']='discarded_scene_changed_during_preflight';record(entry);continue
                points=[ground_pixel(p.pixel_xy,confirmed['depth'],confirmed['camera_info'],
                    source_stamp_s=confirmed['source_stamp_s'],epoch=confirmed['epoch'],max_age_s=1) for p in points]
                entry['grounding_after_preflight']=[p.__dict__ for p in points]
                execution_epoch=_identity(confirmed)
                execution=skills.run(action,*points)
                entry['execution']={'status':execution['status'],'job_ids':[j['job_id'] for j in execution['jobs']]}
                # Wait for a post-execution acquisition, not a cached pre-release image.
                finished=time.monotonic()
                while camera.latest()['source_stamp_s']<=finished:time.sleep(.05)
                after=camera.latest()
                after_path=output/f'{step:03d}-after.jpg';_save_frame(after,after_path)
                entry['after_image']=str(after_path)
                entry['verification']=verify_block_destination(after,decision['object_label'],points[1],
                    destination_region=entry.get('destination_region') if action=='pick_place' else None)
                entry['verification']['evidence_refs']=[str(after_path)]
                if entry['verification']['verdict']=='supported':
                    verified_stages.add(decision['stage_id'])
            except Exception as exc:
                executor.stop()
                entry['execution']={'status':'failed','error':str(exc),'completed_job_ids':[j['job_id'] for j in skills.trace]}
            record(entry)
            if entry.get('execution',{}).get('status')=='failed':
                report['status']='execution_failed';break
        report['verified_stages']=sorted(verified_stages)
        report['segment_progress']={'active_index':segment_index,'count':segment_count(demo)}
    except BaseException as exc:
        report['status']='error';report['error']=f'{type(exc).__name__}: {exc}'
        if 'entry' in locals() and entry not in history:
            entry['outcome']='rejected_before_execution';entry['error']=report['error'];record(entry)
        try:executor.stop()
        except Exception:pass
        if isinstance(exc,KeyboardInterrupt):report['status']='cancelled'
    finally:
        try:worker.close()
        except Exception as exc:report['worker_shutdown_error']=str(exc)
        if camera.thread.ident is not None:
            try:camera.close()
            except Exception as exc:report['recording_error']=str(exc)
        from .report import write_report
        report['elapsed_s']=time.monotonic()-started_monotonic
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_report(report,output)
    return report
