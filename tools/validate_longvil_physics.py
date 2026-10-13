"""Unscored contact and task-feasibility checks through the normal remote MCP."""
import argparse
import asyncio
import base64
from datetime import timedelta
import json
import hashlib
from pathlib import Path
import shlex

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from longvil_physics_execute import Player
from longvil_physics_score import score_episode
from plan_protocol import strict_program_actions


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    ssh=['-T','-S',str(args.socket),'-p',str(args.port),'-o','BatchMode=yes',args.host]
    source=Path(args.helper).parents[1]
    pythonpath=str(source/'piperx-cli/src')+':'+str(source/'piper-lab/src')
    async def remote(operation):
        command=shlex.join(['env','PYTHONPATH='+pythonpath,args.python,args.helper,operation,'--root',args.profile])
        process=await asyncio.create_subprocess_exec('ssh',*ssh,command,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        out,err=await asyncio.wait_for(process.communicate(),60)
        if process.returncode:raise RuntimeError(err.decode()[-1500:])
        return json.loads(out)
    data=json.loads(args.annotation.read_text()) if args.annotation else {
        'object_list':['red cube','blue cube','tray'],
        'code':["p=getpos('red cube')","moveto(p)","pick()","q=getpos('tray','into')","moveto(q)","place()"]}
    report={'scope':'unscored physics/SDK feasibility; no model requests','status':'starting',
            'scene_sha256':(args.metadata or {}).get('scene_sha256'),
            'executor_sha256':hashlib.sha256(Path(__file__).with_name('longvil_physics_execute.py').read_bytes()).hexdigest()}
    with (args.output/'events.jsonl').open('w') as log,(args.output/'stderr.log').open('w') as err:
        def event(row):
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            if row['kind']=='sdk':print(row['name'],row['result'].get('status'),flush=True)
        command=shlex.join(['env','MUJOCO_GL=egl','PYOPENGL_PLATFORM=egl','PYTHONPATH='+pythonpath,args.python,args.helper,
                           'start','--root',args.profile,'--scene',args.scene])
        params=StdioServerParameters(command='ssh',args=[*ssh,command])
        try:
            async with stdio_client(params,errlog=err) as (read,write):
                async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=210)) as session:
                    initialization=await session.initialize();report['cli_version']=initialization.serverInfo.version
                    result=await session.call_tool('robot_connect',{})
                    if result.isError:raise RuntimeError(json.dumps(result.structuredContent))
                    before=await remote('score')
                    async def capture(index):
                        result=await session.call_tool('camera_observe',{'max_frames':1})
                        if result.isError:raise RuntimeError('Camera observation failed')
                        (args.output/f'frame-{index}-metadata.json').write_text(json.dumps(result.structuredContent,ensure_ascii=False))
                        for c in result.content:
                            if c.type=='image':(args.output/f'frame-{index}.jpg').write_bytes(base64.b64decode(c.data))
                    await capture('initial')
                    player=Player(session,lambda:remote('score'),capture,event,args.output.name,
                                  args.metadata if args.metadata else {'base_from_dataset':before['base_from_dataset']})
                    report.update(await player.execute(data['code'],data['object_list']))
                    after=await remote('score');report.update(before=before,after=after,status='executed')
                    report['physics_score']=score_episode(strict_program_actions(data['code'],data['object_list']),
                                                         report['realized_steps'],before,after)
        except Exception as exc:
            report.update(status='error',error_type=type(exc).__name__,error=str(exc)[:1800])
        finally:
            report['cleanup']=await remote('check')
            (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'status':report['status'],'physics_success':report.get('physics_score',{}).get('success'),
        'all_steps':report.get('all_requested_steps_executed'),'cleanup':report['cleanup']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['host','python','helper','profile','scene']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--port',type=int,default=54979)
    parser.add_argument('--socket',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotation',type=Path);parser.add_argument('--scene-metadata',type=Path)
    args=parser.parse_args();args.metadata=json.loads(args.scene_metadata.read_text()) if args.scene_metadata else None
    asyncio.run(run(args))
