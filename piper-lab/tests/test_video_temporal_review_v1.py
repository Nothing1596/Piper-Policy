from fractions import Fraction
import av
import numpy as np
from piperlab.video.source import VideoSource
from piperlab.video.candidates import build_candidates


def write_frames(path,times,colors):
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=25)
        stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
        stream.time_base=Fraction(1,1000);stream.codec_context.time_base=Fraction(1,1000)
        stream.codec_context.max_b_frames=0
        for stamp,color in zip(times,colors):
            frame=av.VideoFrame.from_ndarray(np.full((64,64,3),color,np.uint8),format='rgb24')
            frame.pts=stamp;frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def test_variable_frame_times_and_brief_event_are_recoverable_by_pts(tmp_path):
    path=tmp_path/'vfr.mp4';times=[0,40,50,120,160,500,540,1000]
    write_frames(path,times,[10,10,240,10,10,10,10,10])
    source=VideoSource(path)
    decoded=list(source.frames())
    assert np.allclose([f.timestamp_s for f in decoded],np.array(times)/1000)
    assert abs(source.at(.13).timestamp_s-.16)<1e-9
    # The 10 ms event can be absent from coarse sampling but is retained in
    # the original source and its exact keyframe-backed index.
    assert source.at(.05).image.mean()>200
    assert source.frame_index()['frames'][2]['timestamp_s']==.05


def test_cut_boundary_survives_the_real_decode_and_candidate_pipeline(tmp_path):
    path=tmp_path/'cut.mp4'
    times=list(range(0,2000,100))
    write_frames(path,times,[10 if stamp<1300 else 240 for stamp in times])
    result=build_candidates(str(path),str(tmp_path/'candidates'))
    cut=[f for f in result['candidates'] if 'scene_cut_after' in f['reasons']]
    before=[f for f in result['candidates'] if 'scene_cut_before' in f['reasons']]
    assert any(abs(f['timestamp_s']-1.3)<1e-9 for f in cut)
    assert any(abs(f['timestamp_s']-1.2)<1e-9 for f in before)
