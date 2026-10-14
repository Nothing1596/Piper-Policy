"""Latest-only camera/CV worker with PTS-based recording and bounded memory."""
from fractions import Fraction
from pathlib import Path
import threading
import time
from .grounding import scene_unchanged
from .scene import measured_candidates


class ObservationStream:
    def __init__(self,executor,output_dir,period_s=.5):
        self.executor=executor
        self.output=Path(output_dir)
        self.period=period_s
        self.lock=threading.Lock()
        self.stop_event=threading.Event()
        self.ready=threading.Event()
        self.value=None
        self.error=None
        self.version=0
        self.thread=threading.Thread(target=self._run,name='live-camera-cv',daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(35):
            raise RuntimeError('camera_start_timeout')
        return self.latest()

    def latest(self,max_age_s=1.5):
        with self.lock:
            if self.error:
                raise RuntimeError('camera_worker_failed: '+self.error)
            if self.value is None or time.monotonic()-self.value['source_stamp_s']>max_age_s:
                raise RuntimeError('camera_observation_stale')
            return self.value

    def _run(self):
        import av
        container=None
        stream=None
        first_stamp=None
        last_pts=-1
        try:
            container=av.open(str(self.output/'simulation.mp4'),'w')
            while not self.stop_event.is_set():
                started=time.monotonic()
                current=self.executor.capture()
                current['candidates']=measured_candidates(current)
                with self.lock:
                    identity=lambda x:(x.get('instance_id'),x.get('connection_epoch'),x['epoch'])
                    if self.value is not None and (identity(current)!=identity(self.value) or not scene_unchanged(self.value['rgb'],current['rgb'])):
                        self.version+=1
                    current['state_version']=self.version
                    self.value=current
                self.ready.set()
                if stream is None:
                    h,w=current['rgb'].shape[:2]
                    stream=container.add_stream('libx264',rate=10)
                    stream.width=w;stream.height=h;stream.pix_fmt='yuv420p'
                    stream.time_base=Fraction(1,1000)
                    stream.codec_context.time_base=Fraction(1,1000)
                    stream.codec_context.max_b_frames=0
                    first_stamp=current['source_stamp_s']
                frame=av.VideoFrame.from_ndarray(current['rgb'],format='rgb24')
                pts=round((current['source_stamp_s']-first_stamp)*1000)
                if pts<=last_pts:
                    raise RuntimeError('camera_nonmonotonic_timestamp')
                frame.pts=pts;frame.time_base=Fraction(1,1000);last_pts=pts
                for packet in stream.encode(frame):container.mux(packet)
                self.stop_event.wait(max(0,self.period-(time.monotonic()-started)))
        except Exception as exc:
            with self.lock:self.error=str(exc)
            self.ready.set()
        finally:
            if container is not None:
                try:
                    if stream is not None:
                        for packet in stream.encode():container.mux(packet)
                except Exception as exc:
                    with self.lock:self.error=str(exc)
                finally:container.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=36)
        if self.thread.is_alive():
            raise RuntimeError('camera_worker_shutdown_timeout')
        if self.error:raise RuntimeError('camera_worker_failed: '+self.error)
