"""Fresh agy sessions, same existing simulation, fixed vs fixed+wrist camera.

Private grading stays in the coordinator. The decision process receives only
public messages and hash-allowlisted image files through the agy adapter.
"""
import argparse
import asyncio
import json
from pathlib import Path
import random
from types import SimpleNamespace

from agy_online_decider import AgyDecider, save
import run_longvil_online as online


def write_comparison(output):
    rows=['# Gemini 单/双相机个案对照', '',
          '同一 complex 示例、plan/skill 开启、独立新会话。仅物理仿真；各一次运行不支持显著性或泛化结论。', '',
          '| 条件 | 状态 | 目标关系 | 撤离 | 撤离后稳定 | 完整完成 | 决策响应 | 模型生成 | 看图次数 | completion tokens | 秒 |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in ('single','dual'):
        path=output/name/'report.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        values=[name]+[r.get(k) for k in ('status','target_relations_success','retreat_verified','stable_after_retreat',
                  'task_complete','decision_rounds','model_invocations','confirmed_image_views','completion_tokens','elapsed_s')]
        rows.append('| '+' | '.join('未知' if v is None else str(round(v,2)) if isinstance(v,float) else str(v) for v in values)+' |')
    rows+=['','实际费用未由 CLI 提供，记为未知。thinking 已包含在 completion 中。双相机顺序采集，非同步。',
           '成功判断来自各轮 physics-score.private.json；原始反馈、拒绝、动作链及逐轮 CLI 回执均保留。']
    (output/'comparison.zh.md').write_text('\n'.join(rows)+'\n')


async def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    conditions=['mujoco-overview','all']
    random.Random(config['seed']).shuffle(conditions)
    save(output/'config.json', config)
    save(output/'schedule.json',conditions)
    for condition in conditions:
        name='single' if condition=='mujoco-overview' else 'dual'
        exploration=config.get('exploration',False)
        decider=AgyDecider(output/(name+'-decision'), model=config['model'],exploration=exploration)
        args=SimpleNamespace(
            output=output/name, annotation=Path(config['annotation']), evidence=Path(config['evidence']),
            socket=Path(config['socket']),host=config['host'],port=config['port'],
            remote_root=config['remote_root'],remote_python=config['remote_python'],
            scene=config['scene'],task=config['task'],object_list=json.dumps(config['objects']),
            effort='high',plan='on',skill='on',decision_policy='standard',call_completion_cap=65536,
            interface_smoke=False,contact_smoke=False,native_probe=False,
            max_rounds=config.get('max_rounds',80),max_tools=config.get('max_tools',120),
            max_seconds=config.get('max_seconds',1800),completion_budget=config.get('completion_budget',65536),exploration=exploration,
            camera_condition=condition,model_decider=decider,requested_model=config['model'],strict_completion=True,plan_envelope=True)
        save(output/'state.json',{'status':'running','condition':name})
        try:
            await online.main(args)
        finally:
            await decider.close()
        report=json.loads((args.output/'report.json').read_text())
        receipts=[json.loads(p.read_text()) for p in sorted(decider.root.glob('turn-*-receipt.json'))]
        audits=[json.loads(line) for line in decider.audit_path.read_text().splitlines()]
        views=[r for r in audits if r['phase']=='post' and r['tool']=='view_file' and r['allowed'] and r.get('path') and Path(r['path']).suffix in ('.jpg','.png') and not r.get('error')]
        report.update(model=config['model'],model_identity_scope='Requested agy profile; underlying weights unknown',
                      camera_condition=condition,decision_rounds=report.get('api_responses'),
                      model_invocations=sum(r['phase']=='invocation' for r in audits),
                      confirmed_image_views=len(views),
                      unique_images_viewed=len({r['path'] for r in views}),
                      cli_results=len(list(decider.root.glob('turn-*-result.json'))),
                      runtime_model_aliases=sorted({r['model'] for r in audits if r.get('model')}),
                      provider_usage_source='agy cumulative result deltas',estimated_cny=None,
                      cost_status='Unavailable from agy, never estimated as zero',
                      cli_conversation_id=decider.conversation,cli_conversation_ids=decider.conversations,
                      harness_session_id=decider.context.session_id,context_policy='harness-managed-per-decision',gt_visible_to_decider=False,
                      execution_mode='concurrent model tool batches; coordinated state changes' if exploration else 'one real CLI decision per tool; no scripted action queue',
                      exploration=exploration,completion_budget=args.completion_budget)
        if receipts:
            report.update(completion_tokens=sum(r['usage_delta']['output_tokens'] for r in receipts),
                          reasoning_tokens=sum(r['usage_delta']['thinking_tokens'] for r in receipts))
        results=list(decider.root.glob('turn-*-result.json'))
        report['usage_complete']=(not any(decider.root.glob('turn-*-transport-error.json'))
                                  and len(results)==len(receipts))
        if not report['usage_complete']:
            report['completion_tokens_known']=report.get('completion_tokens',0)
            report['completion_tokens']=None
            report['reasoning_tokens']=None
        save(args.output/'report.json',report)
        write_comparison(output)
        if report['status']=='paused' or report.get('cleanup_error') or report.get('cleanup',{}).get('runtime_remaining'):
            save(output/'state.json',{'status':'paused','condition':name,'report':str(args.output/'report.json')})
            return
    save(output/'state.json',{'status':'complete','conditions':conditions})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    asyncio.run(run(json.loads(args.config.read_text()),args.output))
