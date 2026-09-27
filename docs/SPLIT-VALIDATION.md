# 独立发行包验收 / Standalone release validation

Date: 2026-09-27. Video 0.3.0 / robot 0.6.0. Windows x64, Python 3.12.

## 源码回归 / Source regression

- Upper: **290 passed, 3 skipped**.
- Lower: **48 passed, 5 skipped**. External SDK contract tests skip because that SDK is absent. One upstream Starlette/httpx deprecation warning remains.
- Added checks prevent the video CLI enabling robot tools, reject camera output traversal/overwrite, and verify the robot camera is workspace opt-in.
- The existing combined MCP still exposes both tool sets; its camera implementation now resides in the lower distribution.

## 独立环境 / Independent installations

Each installer creates a fresh virtual environment from its own SHA256-pinned offline wheel closure. Video contains 80 wheels; robot contains 49. Both pass `pip check` and CLI startup. The video interpreter cannot import `piperx_middleware`; the robot interpreter cannot import `piperlab`.

Video checks: actual local candidate extraction, ONNX detector initialization, stdio MCP initialization and tool listing, candidate paging and a 640×480 image return. No robot tools are exposed. No cloud inference was needed or performed for these packaging checks.

Robot checks: fresh MuJoCo root on isolated port 8819, connect through the independent CLI, save RGB 640×480 and depth 480×640, stdio MCP status query and actual image return, stop request, normal executor shutdown. The returned image was inspected and showed the arm, red/blue cubes and green tray. No video tools are exposed.

`tools/validate_split_install.py` reproduces these checks under the installed interpreter. Its work directory must be new. Robot test data includes local runtime credentials and must never be published wholesale. Public receipts contain only validation metadata.

## 结论范围 / Scope

These results establish independent deployment and tool communication. They do not establish new model semantics, autonomous task success, human-video transfer or real-arm execution. Prior combined-release synthetic 6/6 results remain historical; they are not rerun or relabeled as standalone-release task evaluations.

Large model weights, cloud credentials and the vendor hardware SDK are not included. The video wheel retains the `piper-lab` distribution name and legacy upper modules; `piper-video` is the restricted video entry point. The standalone robot includes its own camera tool and simulator.

## 构建 / Build

Build both project wheels, then use `tools/build_split_packages.py` with a local dependency wheel depot, freshly built project wheels, the prior public kit as the resource template, and a new output directory. Dependency resolution uses `--no-index`; private runtime files are never copied. Each ZIP includes its own bilingual guide, notices, requirements lock and SHA256 manifest. Installed virtual environments and validation work directories are excluded from ZIPs.
