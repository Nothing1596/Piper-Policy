# Detector source and export provenance

The installation ZIP includes models/yolo11n.onnx, derived from Ultralytics YOLO11n. Its exact binary hash is listed in the ZIP's SHA256.json. The export environment used Ultralytics **8.4.163**. This detector is a separate perception component; it is not the vision-language model used to interpret a demonstration.

- Upstream source: https://github.com/ultralytics/ultralytics
- Pinned Python source distribution: `ultralytics-8.4.163.tar.gz`, supplied as a separate asset with release v0.2.1.
- License: the applicable upstream AGPL-3.0 or a separately obtained commercial license. The bundle retains the upstream notice; publication does not confer a commercial-license exemption.
- Project-side detection, tracking and preprocessing: `piper-lab/src/piperlab/perception/detector_onnx.py` and its associated tests.

The source archive is offered alongside the detector-containing package so recipients can inspect the exporting implementation. It is not installed by install.ps1 and does not contain project API keys or private runtime data.

Export settings should be read from the actual ONNX metadata and the project adapter. Re-exporting on another toolchain can change the binary hash; compare behavior and metadata rather than claiming byte-identical weights. Training the YOLO model from scratch is not part of this project.

The shipped ONNX metadata was checked: version 8.4.163, opset 17, input 640×640, batch 1, dynamic=False, simplify=False, no embedded NMS. An equivalent export invocation, after obtaining the corresponding upstream YOLO11n checkpoint, is:

```python
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='onnx', imgsz=640, batch=1,
                        dynamic=False, simplify=False, opset=17)
```

This is a reproduction recipe from recorded metadata, not a claim that a fresh export is bit-identical. Dependency versions and checkpoint identity also affect reproducibility.
