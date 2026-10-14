"""ACT on live ROS observations. Default is shadow only; --execute is explicit."""
import argparse
import base64
import io
import json
import os
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .adapter import PiperBridgeAdapter, _stamp_ns, StaleObservationError
from .learning import _resolve_cuda_visible_devices, _resolve_pretrained_model_dir
from .safety import load_config, commissioning_errors

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',default='config/hardware.yaml'); parser.add_argument('--url',default='ws://127.0.0.1:9090')
    parser.add_argument('--output',required=True); parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); config=load_config(args.config)
    if args.execute and commissioning_errors(config):
        raise RuntimeError('Hardware is not commissioned: '+', '.join(commissioning_errors(config)))
    os.environ['CUDA_VISIBLE_DEVICES']=_resolve_cuda_visible_devices(config['gpu_uuid'])
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    checkpoint=_resolve_pretrained_model_dir(args.checkpoint)
    policy=ACTPolicy.from_pretrained(checkpoint).to('cuda').eval()
    if policy.config.n_action_steps != 1: raise RuntimeError('Live v1 requires n_action_steps=1')
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
    height,width=policy.config.input_features['observation.images.rgb'].shape[-2:]
    lock=threading.Lock(); latest=[None]
    def image_callback(msg):
        with lock: latest[0]=msg
    adapter=PiperBridgeAdapter(args.url,owner='policy')
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    def infer(state, message):
        payload=message['data']; payload=base64.b64decode(payload) if isinstance(payload,str) else bytes(payload)
        rgb=np.asarray(Image.open(io.BytesIO(payload)).convert('RGB').resize((width,height))).copy()
        batch={'observation.state':torch.from_numpy(state.vector).unsqueeze(0).cuda(),
               'observation.images.rgb':torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).cuda().float()/255.}
        with torch.inference_mode(): return post(policy.select_action(pre(batch))).cpu().numpy().reshape(-1).tolist()
    image_topic=None
    try:
        adapter.connect(); image_topic=adapter._make_topic('/lab/rgb','sensor_msgs/CompressedImage'); image_topic.subscribe(image_callback)
        deadline=time.monotonic()+10
        while latest[0] is None and time.monotonic()<deadline: time.sleep(.02)
        if latest[0] is None: raise RuntimeError('No camera stream')
        state=adapter.wait_observation(); infer(state,latest[0]); torch.cuda.synchronize(); policy.reset()
        # Warm-up precedes acquisition, so first-use CUDA initialization cannot trip an armed robot.
        if args.execute: adapter.acquire(); adapter.arm()
        end=time.monotonic()+args.seconds; next_tick=time.monotonic()
        with output.open('x',encoding='utf-8') as log:
            while time.monotonic()<end:
                state=adapter.latest_observation()
                with lock: msg=latest[0]
                image_stamp=_stamp_ns(msg)/1e9
                if not -.05 <= time.time()-image_stamp <= .2: raise StaleObservationError('Camera expired')
                start=time.perf_counter(); target=infer(state,msg); torch.cuda.synchronize()
                latency=time.perf_counter()-start
                if latency>.2 or time.time()-min(state.stamp_ns/1e9,image_stamp)>.2:
                    raise StaleObservationError('Inference or source observation expired; output discarded')
                if args.execute: adapter.send_command(target)
                log.write(json.dumps({'timestamp':time.time(),'observation_stamp_ns':state.stamp_ns,'image_stamp_ns':_stamp_ns(msg),
                                     'latency_ms':latency*1000,'target':target,'executed':args.execute,'synthetic':config['mode']=='simulation'})+'\n'); log.flush()
                next_tick=max(next_tick+1/config['control_hz'],time.monotonic())
                time.sleep(max(0,next_tick-time.monotonic()))
    finally:
        if image_topic is not None: image_topic.unsubscribe()
        adapter.close()

if __name__=='__main__': main()
