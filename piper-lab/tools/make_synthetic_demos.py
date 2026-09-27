"""Deterministic pipeline fixtures. These contain NO human demonstration."""
from fractions import Fraction
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    specs={
        'transfer':{'task':'Move the demonstrated block to the green region.','order':['red']},
        'order_red_blue':{'task':'Arrange both blocks in the green region in the demonstrated order.','order':['red','blue']},
        'order_blue_red':{'task':'Arrange both blocks in the green region in the demonstrated order.','order':['blue','red']},
        'push':{'task':'Push the demonstrated block along the shown direction.','order':['red']}}
    for name,spec in specs.items():
        path=output/(name+'.mp4')
        with av.open(str(path),'w') as container:
            stream=container.add_stream('libx264',rate=20)
            stream.width=640;stream.height=480;stream.pix_fmt='yuv420p'
            for i in range(120):
                t=i/20
                frame=Image.new('RGB',(640,480),(225,226,224));draw=ImageDraw.Draw(frame)
                draw.rectangle((390,160,570,390),fill=(40,145,80),outline=(20,65,30),width=8)
                positions={'red':(155,200),'blue':(185,330)}
                for j,color in enumerate(spec['order']):
                    progress=min(1,max(0,(t-(1+j*2))/1.5))
                    x,y=positions[color]
                    tx,ty=(470,210+j*105)
                    if name=='push':tx,ty=(330,200)
                    positions[color]=(x+(tx-x)*progress,y+(ty-y)*progress)
                for color,(x,y) in positions.items():
                    draw.rectangle((x-18,y-18,x+18,y+18),fill=(205,35,35) if color=='red' else (35,70,205),outline='black',width=2)
                draw.text((15,15),'SYNTHETIC PIPELINE FIXTURE - NO HUMAN',fill='black')
                video=av.VideoFrame.from_ndarray(np.array(frame),format='rgb24');video.pts=i;video.time_base=Fraction(1,20)
                for packet in stream.encode(video):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        spec.update(video=str(path.resolve()),provenance='synthetic_2d_pipeline_fixture',human=False)
    (output/'manifest.json').write_text(json.dumps(specs,indent=2),encoding='utf-8')

if __name__=='__main__':
    import sys
    generate(sys.argv[1])
