# Piper combined CLI 0.2.1

[中文](README.md) | [English](README.en.md)

Windows x64 / Python 3.12. Includes upper-level piper-lab 0.2.1 and lower-level piperx-middleware 0.5.0 in one isolated environment.

Start with [START-HERE.md](START-HERE.md) and the [detailed deployment guide](docs/NEW-MACHINE-GUIDE.md), both in Chinese. The [repository English README](https://github.com/Nothing1596/Piper-Policy/blob/main/README.en.md) provides an English overview and quick start.

Extract to a permanent directory and run PowerShell there:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper.cmd doctor
.\piper.cmd --help
```

The installer checks SHA256 hashes and installs bundled dependency wheels offline. Python 3.12 x64 is required separately. Installation does not start a model, connect hardware, or change firmware. Do not move an installed virtual environment to another machine; install again from the ZIP.

Start simulation in terminal A:

```powershell
.\piper.cmd sim start --root work\sim --port 8808 --seed 200
```

Connect from terminal B in the same directory:

```powershell
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 connect
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 status
```

Then use a model profile with `piper policy run`, or connect `piper mcp` to an agent. `simulation_observe` is included in 0.2.1; no camera-tool patch is required. CLI groups include demo/video, models, policy, sim, eval, device, lab and mcp.

Vision model weights, LM Studio, agent clients and cloud accounts are separate prerequisites. Copy a profile to your workspace after installation, edit the model ID/endpoint, and provide keys through environment variables. Run `models probe` before a full task. No private videos, credentials, robot tokens or VLM weights are bundled.

The autonomous visual policy is validated only for its bounded MuJoCo task scope. Real-hardware commissioning and arbitrary-object transfer are not established by simulation results. See [release notes](docs/RELEASE-NOTES.md), [validation receipt](PACKAGE-VALIDATION.json) and [third-party notices](THIRD-PARTY.md).
