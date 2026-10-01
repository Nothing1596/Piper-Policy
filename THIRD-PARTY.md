# Third-party components

This repository and its release bundle preserve the licenses of their respective components; publication is not a new license grant. Dependency wheels retain upstream metadata and license files. Robot asset provenance and its MIT notice are in piperx-cli/assets/mujoco; additional notices are in piperx-cli/LICENSES. The installation ZIP carries its own licenses/ directory.

YOLO11n ONNX is derived from Ultralytics YOLO11 (https://github.com/ultralytics/ultralytics, export environment 8.4.163). It is subject to its upstream AGPL-3.0 or applicable commercial license; packaging does not grant a commercial redistribution exemption. For redistribution, retain notices and meet the applicable source/license obligations. The included detector hash is in SHA256.json.

No private videos, hardware tokens or VLM model weights are included. Any separately obtained VLM weights retain their own license.

The detector is distributed in the complete release, not in the Git tree. See docs/DETECTOR-SOURCE.md for the pinned upstream source and export information. No repository-wide license has been selected for all original project code.

## Separately supplied GPT-Policy baseline

The optional `gpt_policy_piper` integration targets cheng-haha/GPT-Policy commit
`ab970d88bc5570d80a7b4f3e8d3ad97ebe65a007`. GPT-Policy source is not included in
this repository or its adapter wheel. Its current LICENSE is review/evaluation
only, with no selected redistribution/commercial license. Obtain permission
before redistributing its separately prepared checkout. See
`docs/GPT-POLICY-WINDOWS-SIM.md` for the boundary and verification limits.
