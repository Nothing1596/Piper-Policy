# Windows 一键安装 / One-click installation

[中文 README](../README.md) · [English README](../README.en.md)

适用独立视频包和下位机包，不适用旧联合包。只负责安装与命令注册，不启动机器人或模型。

## 使用 / Use

1. 将完整离线包解压到准备长期保留的目录。
2. 包内已有 `Setup.cmd` 时直接双击。较早的 video 0.3.0 / robot 0.6.0 ZIP 没有它：下载 [一键安装补充包](https://github.com/Nothing1596/Piper-Policy/releases/tag/installer-v1)，将里面的三个脚本解压到完整包根目录，与 `SHA256.json`、`wheels` 同级。
3. 双击 `Setup.cmd`。完成后重新打开终端，运行 `piper-video --help` 或 `piper-robot --help`。

Extract the complete standalone kit into a permanent directory. Double-click `Setup.cmd`. For the original video 0.3.0 / robot 0.6.0 ZIP, extract the small installer add-on beside `SHA256.json` first. Open a new terminal after installation. The add-on is not a replacement for the full dependency kit.

安装器自动完成 / Automated steps:

- 识别视频包或下位机包，核对包内文件 SHA256。
- 检测 Python 3.12 x64；缺少时通过 winget 安装用户级 Python。只有此步骤需要网络；没有 winget 时给出明确提示，不擅自换 Python 版本。
- 创建或复用包目录的 `.venv`，从本地 wheels 离线安装依赖并运行 `pip check`、CLI 帮助检查。
- 在 `%USERPROFILE%\.piper\bin` 创建专用启动器，并将该目录加入**用户 PATH**；保留已有 PATH 条目，不修改系统 PATH 或永久执行策略。
- 保存注册回执和原 PATH 备份。重复运行不会重复追加 PATH，已有可用环境可以复用；失败时不会自动删除环境。

Python discovery, environment creation, offline dependency checks and per-user CLI registration are automatic. Python installation uses [Microsoft's documented winget install interface](https://learn.microsoft.com/en-us/windows/package-manager/winget/install), with user scope and noninteractive agreement acceptance. Existing Python is reused. The missing-Python download branch requires network access; do not infer that it was exercised on a host where Python already exists.

## 命令行选项 / Options

```powershell
# 指定 Python；无需安装其他 Python / Use an existing interpreter
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -PythonExe C:\Python312\python.exe

# 离线模式：不自动下载 Python / Never download Python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -NoPythonInstall

# 只安装到包内，不注册 PATH / Install without PATH registration
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -NoRegister

# 脚本与包不在同一目录 / Explicit package location
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -PackageRoot D:\PiperRobot
```

不要删除或移动已注册的包目录；启动器引用其中的解释器。迁移到新目录后重新执行安装器更新注册。不要复制 `.venv` 到新机器。安装不配置模型密钥、CAN、代理客户端或 GPU。

Do not move/delete the registered package directory. Reinstall from an extracted kit on another machine. Registration does not configure API credentials, CAN, MCP clients or GPU placement.

## 取消命令注册 / Unregister

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\unregister-cli.ps1 -Component video
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\unregister-cli.ps1 -Component robot
```

只移除对应启动器；当没有其他 CMD 启动器时移除这一条用户 PATH。保留包目录、Python、数据以及其他 PATH 条目。若启动器被手工改动，脚本会保留它并提示检查。重开终端后生效。This unregisters commands only; it does not delete Python or package data.

若安装完成但旧终端找不到命令，先关闭该终端和其宿主应用，再新开 PowerShell。可用 `Get-Command piper-video` / `Get-Command piper-robot` 检查实际解析位置，避免同名命令遮蔽。
