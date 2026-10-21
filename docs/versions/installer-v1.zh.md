# installer-v1：旧版安装补充包说明

[返回版本导航](README.zh.md) · [正式下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/installer-v1) · [详细选项](../ONE-CLICK-INSTALL.md)

## 适用范围

这是 2026-09-27 发布的约 7 KB 脚本附件 `piper-one-click-installer-v1.zip`，不是完整机器人或视频程序。只用于已下载的 Video 0.3.0 / Robot 0.6.x 完整独立包，不适用于 Robot 0.7.1 和开发交互包。

它提供 Setup.cmd、安装/注册脚本和撤销注册脚本。完整程序、依赖 wheels、模型及硬件驱动不在此补充包中；没有先下载完整包时不能独立安装成功。

## 安装操作

1. 把完整 Video 或 Robot 旧包解压到长期保留的目录。
2. 核对补充包 SHA256SUMS.txt，把三个脚本放到完整包根目录，与 SHA256.json 和 wheels 同级。
3. 双击 Setup.cmd，或在该目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1
```

完成后关闭旧终端并打开新 PowerShell：

```powershell
Get-Command piper-robot
piper-robot --help
```

视频包把命令替换为 piper-video。检查解析路径是否指向刚安装的包，避免被其他同名启动器遮蔽。

## 自动完成的功能

| 步骤 | 实际行为 |
|---|---|
| 识别包和校验 | 检查组件及 SHA256，不静默改包内容 |
| 识别 Python | 要求 Python 3.12 x64；缺少时可通过 winget 安装用户级 Python |
| 安装环境 | 包内 `.venv`，从 wheels 离线安装并 pip check |
| 注册命令 | 在用户 `.piper/bin` 创建启动器，追加用户 PATH，不修改系统 PATH |
| 留存与重复运行 | 保存注册回执和 PATH 备份，不重复追加 PATH，不自动删除失败环境 |

缺少 Python 的下载步骤需要网络和可用 winget；不能将其描述为全流程完全离线。安装不启动机器人、不连模型，不配置 CAN、API key 或 GPU。

## 常用选项和撤销

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -NoPythonInstall
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -NoRegister
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install-cli.ps1 -PythonExe C:\Python312\python.exe
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\unregister-cli.ps1 -Component robot
```

上面是替代调用方式，不需全部依次执行。`-NoPythonInstall` 禁止自动下载；`-NoRegister` 只安装包内环境；撤销支持 robot/video，只移除对应启动器及可独立移除的 PATH 项，保留程序、数据与 Python。

启动器引用原包的绝对解释器路径，因此不能注册后随意移动或删除目录；迁移应在新目录重新安装/注册。退出旧终端再检查新 PATH。

历史验证覆盖已有 Python 安装、重复注册不重复 PATH、隔离撤销及新 PowerShell 命令解析；没有在缺 Python 的主机上执行下载分支。本次没有改注册表或运行安装器。
