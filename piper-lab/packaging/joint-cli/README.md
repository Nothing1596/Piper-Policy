# Piper 联合 CLI 0.2.1

[中文](README.md) | [English](README.en.md)

Windows x64 / Python 3.12；piper-lab 0.2.1 + piperx-middleware 0.5.0。

**首次部署请阅读 [START-HERE.md](START-HERE.md)，随后按 [新机器部署向导](docs/NEW-MACHINE-GUIDE.md) 操作。**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper.cmd doctor
.\piper.cmd --help
```

统一 CLI 包含 demo/video、models、policy、sim、eval、device、lab、mcp。0.2.1 已集成 simulation_observe，无需另装相机工具补丁。

安装依赖离线完成；Python 3.12 x64、外部代理客户端、本地模型服务/权重或云 API 账户需自行准备。安装不会启动模型或连接机械臂。新机器需重新安装，不复制 .venv。

- [完整部署、模型配置、MCP 接入、仿真和排障](docs/NEW-MACHINE-GUIDE.md)
- [模型协议与接入说明](docs/model-providers-and-harnesses.md)
- [本版本变化和验证边界](docs/RELEASE-NOTES.md)
- [本包安装验证](PACKAGE-VALIDATION.json)
- [第三方内容与许可](THIRD-PARTY.md)

包不包含真实人类视频、密码、API key、机器人 token 或 VLM 权重。配置示例在 profiles/；安装完成后复制到工作目录再修改。
