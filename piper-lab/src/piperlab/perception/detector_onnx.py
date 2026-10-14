"""YOLO11 ONNX candidate scorer, optional CPU inference; no torch imports."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import numpy as np
from .detect import Detection


class OnnxDetector:
    name = "yolo11n_onnx"

    def __init__(self, model_path, *, threshold=.1, nms_iou=.45, max_candidates=128):
        import onnxruntime as ort
        if not 0 < threshold < 1 or not 0 < nms_iou < 1 or max_candidates < 1:
            raise ValueError("Invalid detector parameters")
        path = Path(model_path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta.get("names", "{}"))
        self.threshold, self.nms_iou, self.max_candidates = threshold,nms_iou,max_candidates
        self.last_report = None

    def score(self, image, *, source_id, sequence, source_s):
        import cv2
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError("Detector expects RGB uint8 HWC")
        h,w = image.shape[:2]
        scale = min(self.size/h,self.size/w)
        nh,nw = int(round(h*scale)),int(round(w*scale))
        y,x = (self.size-nh)//2,(self.size-nw)//2
        padded = np.full((self.size,self.size,3),114,np.uint8)
        padded[y:y+nh,x:x+nw] = cv2.resize(image,(nw,nh))
        tensor = np.ascontiguousarray(padded.transpose(2,0,1)[None], dtype=np.float32)/255
        output = np.asarray(self.session.run(None,{self.input_name:tensor})[0])
        if output.ndim!=3 or output.shape[0]!=1:
            raise ValueError("Unsupported YOLO output; export raw detection head without NMS")
        rows = output[0]
        if rows.shape[0] < rows.shape[1]:
            rows = rows.T
        if rows.shape[1] < 5 or not np.isfinite(rows).all():
            raise ValueError("Malformed YOLO output")
        scores = rows[:,4:].max(axis=1)
        classes = rows[:,4:].argmax(axis=1)
        indices = np.flatnonzero(scores>=self.threshold)
        selected, drops = [], []
        # Class-aware NMS; results are CPU predictions, not independent evidence.
        for label in np.unique(classes[indices]):
            group = indices[classes[indices]==label]
            boxes = [[float(rows[i,0]-rows[i,2]/2),float(rows[i,1]-rows[i,3]/2),
                      float(rows[i,2]),float(rows[i,3])] for i in group]
            kept = np.asarray(cv2.dnn.NMSBoxes(boxes,scores[group].tolist(),self.threshold,self.nms_iou)).reshape(-1)
            keep_set = {int(group[k]) for k in kept}
            selected.extend(keep_set)
            drops.extend({"index":int(i),"reason":"nms_overlap"} for i in group if i not in keep_set)
        selected.sort(key=lambda i:float(scores[i]),reverse=True)
        if len(selected)>self.max_candidates:
            # Explicit budget failure; caller can change configuration and retry.
            raise ValueError(f"detector_budget: {len(selected)} > {self.max_candidates}")
        result=[]
        for i in selected:
            cx,cy,bw,bh = rows[i,:4]
            x1,y1 = max(0.,float((cx-bw/2-x)/scale)),max(0.,float((cy-bh/2-y)/scale))
            x2,y2 = min(float(w),float((cx+bw/2-x)/scale)),min(float(h),float((cy+bh/2-y)/scale))
            if x2<=x1 or y2<=y1:
                drops.append({"index":int(i),"reason":"outside_image"});continue
            label_id = int(classes[i])
            label = self.names.get(label_id,str(label_id)) if isinstance(self.names,dict) else self.names[label_id]
            result.append(Detection(source_id=source_id,sequence=sequence,source_s=source_s,
                scorer=self.name,label=label,score=float(scores[i]),bbox_xyxy=(x1,y1,x2,y2),
                centroid_xy=((x1+x2)/2,(y1+y2)/2),area_px=(x2-x1)*(y2-y1),
                detail={"class_id":label_id,"model_sha256":self.sha256}))
        self.last_report={"raw_predictions":len(rows),"below_threshold":int((scores<self.threshold).sum()),
                          "selected":len(result),"dropped":drops}
        return result


class ObjectTracker:
    """Track only regularly sampled detection frames; reset at source gaps."""
    def __init__(self, sample_hz=2):
        from trackers import ByteTrackTracker
        self._factory = ByteTrackTracker
        self.tracker = ByteTrackTracker()
        self.sample_hz = sample_hz
        self.last_stamp = None
        self.epoch = 0

    def reset(self):
        self.epoch+=1;self.tracker=self._factory();self.last_stamp=None

    def update(self, detections, source_s):
        import supervision as sv
        if self.last_stamp is not None and (source_s<=self.last_stamp or source_s-self.last_stamp>2/self.sample_hz):
            self.epoch+=1; self.tracker=self._factory()
        self.last_stamp=source_s
        values=sv.Detections(xyxy=np.asarray([d.bbox_xyxy for d in detections],dtype=np.float32).reshape(-1,4),
            confidence=np.asarray([d.score for d in detections],dtype=np.float32),
            class_id=np.asarray([d.detail["class_id"] for d in detections],dtype=int))
        result=self.tracker.update(values)
        return {"epoch":self.epoch,"source_s":source_s,"xyxy":result.xyxy.tolist(),
                "tracker_id":[] if result.tracker_id is None else result.tracker_id.tolist()}


class TrackedDetector:
    """ONNX + ByteTrack proposals; local track IDs have no contact authority."""
    def __init__(self,model_path,sample_hz=2):
        self.detector=OnnxDetector(model_path)
        self.tracker=ObjectTracker(sample_hz)
        from .tracked_entities import TrackedEntities
        self.entities=TrackedEntities()
        self.name='yolo11n_onnx_bytetrack'
        self.sha256=self.detector.sha256
        self.source=None
        self.previous=None
        self.last_report=None

    def score(self,image,*,source_id,sequence,source_s):
        import cv2
        from dataclasses import replace
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(80,60)).astype(float)
        cut=bool(self.previous is not None and np.mean(np.abs(gray-self.previous))>55)
        if source_id!=self.source or cut:self.tracker.reset()
        self.source=source_id;self.previous=gray
        values=self.detector.score(image,source_id=source_id,sequence=sequence,source_s=source_s)
        tracks=self.tracker.update(values,source_s)
        results=[]
        for detection in values:
            x1,y1,x2,y2=detection.bbox_xyxy
            matches=[]
            for box,track_id in zip(tracks['xyxy'],tracks['tracker_id']):
                a,b,c,d=box
                inter=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
                iou=inter/max(1e-9,(x2-x1)*(y2-y1)+(c-a)*(d-b)-inter)
                if track_id>=0 and iou>.5:matches.append((iou,track_id))
            track_id=int(max(matches)[1]) if matches else None
            results.append(replace(detection,detail={**detection.detail,'track_id':track_id,
                'track_epoch':tracks['epoch'],'track_source_id':source_id,'track_scope':'video_local_only'}))
        results,entities=self.entities.update(results,source_id=source_id,epoch=tracks['epoch'],source_s=source_s)
        self.last_report={**self.detector.last_report,'tracker_epoch':tracks['epoch'],'scene_cut_reset':cut,
                          'entity_lifecycle':entities}
        return results
