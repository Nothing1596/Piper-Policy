# 联合版 v0.2.1：功能与使用说明

[返回版本导航](README.zh.md) · [正式下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/v0.2.1) · [该 tag 的原始向导](https://github.com/Nothing1596/Piper-Policy/blob/v0.2.1/docs/NEW-MACHINE-GUIDE.md)

## 这是什么版本

2026-09-27 发布的 Windows x64 / Python 3.12 离线联合包。资产为 `piper-combined-cli-0.2.1-win64-py312.zip`，包含上位 `piper-lab 0.2.1`、下位 `piperx-middleware 0.5.0`、依赖 wheel、MuJoCo 资源和示例。统一入口是包内 `piper.cmd`，不是后来单终端版的斜杠命令界面。

适合复现历史联合视觉策略、示范编译及 MCP 仿真流程。新用户主要做机器人交互时应读 0.7.1 或源码版，不要混用两代安装入口。

## 组件与功能

| 部分 | 命令 | 输入与产物 |
|---|---|---|
| 示范视频 | `piper demo` | MP4→选帧、阶段、引用及 unknown 的示范包 |
| 模型配置 | `piper models` | provider profile、配置检查和单图真实 probe |
| 视觉策略 | `piper policy run` | 当前 RGB-D、目标与可选示范→逐步决策和执行记录 |
| 共享执行器 | `piper device` | 状态、连接、运动、作业及反馈 |
| 仿真 | `piper sim` | MuJoCo 启动与独立评分 |
| 代理工具 | MCP | 示范工具、机器人工具和 `simulation_observe` |

视频解析需要模型服务；直接选帧和机器人手动检查不需要 API key。外部代理自己看图与管线配置 VLM 是两种入口，不会相互继承凭据。

## 安装与首次自检

先下载发布页的校验文件，核对 ZIP；解压到新目录，进入包含 `install.ps1`、`piper.cmd` 和 `wheels` 的目录。PowerShell：

```powershell
py -3.12 -c "import sys,struct; print(sys.version); print(struct.calcsize('P')*8)"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper.cmd doctor
.\piper.cmd --help
```

输出应是 Python 3.12、64 位，安装后 doctor 的 `ok` 为 true。安装器创建包内 `.venv`、校验文件并离线安装；不自动连接硬件或加载模型。不复制其他机器的 `.venv`，也不在安装前修改受校验的包文件。

## 无模型先连接仿真

终端 A 保持运行：

```powershell
.\piper.cmd sim start --root work\sim --port 8808 --seed 200
```

终端 B 使用同一包目录：

```powershell
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 connect
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 status
```

MCP 使用包内解释器，模块为 `piperlab.harness`，参数含 `--workspace`、`--robot-root` 和 `--robot-url`。工作目录用任务专用目录；外部代理通过 `simulation_observe` 获取实际图像，并使用新的 output 目录保存 RGB、深度和标定。MCP 不代替终端 A 的执行器。

## 配模型、看示范并执行

复制包内 provider profile 到 `work/model.json` 后编辑实际端点和模型名；密钥保留为环境变量引用。先完成真实单图 probe，再运行视频解析：

```powershell
.\piper.cmd models show --model-config work\model.json
.\piper.cmd models probe --model-config work\model.json --image examples\probe.jpg --output work\probe-01
.\piper.cmd demo compile --model-config work\model.json --video examples\transfer.mp4 --task "Move the red block into the green tray as demonstrated." --output work\demo-01 --store work\demo-store
.\piper.cmd demo inspect --bundle work\demo-01
.\piper.cmd policy run --model-config work\model.json --task "Move the red block into the green tray." --backend http://127.0.0.1:8808 --token-file work\sim\model.token --output work\policy-01 --max-decisions 12 --demo work\demo-01
```

每次使用新 output 名称，模型失败也保留记录。该视觉策略面向有限彩块/托盘任务，不能把 backend 换成真机地址就认为完成真机部署。任务结束后由操作员独立评分，避免向决策模型暴露私有评分或答案。

## 停止、升级及已知问题

使用 `device ... stop` 停止当前动作，空闲后 `device ... shutdown` 结束执行器。超时先查原请求，不换新编号重放。连接拒绝时核对端口、root 与实例 token；输出已存在则换目录。模型配置 show 成功不代表图片或工具调用成功。

保存的包验证为上位 287 通过/3 跳过；迁移下位源码 44 通过/5 外部 SDK 跳过，并从新 MuJoCo 实例实际回传 RGB-D。历史一次 MCP 抓放案例不是通用成功率，也不是新供应商或真机验收。本次没有重跑这些记录。

迁移到独立版时分开安装 video 和 robot；原素材及记录保留，不在原离线包上混装新依赖。原向导含完整 provider、MCP 和故障说明。
