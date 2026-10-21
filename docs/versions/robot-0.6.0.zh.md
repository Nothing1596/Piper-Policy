# Robot 0.6.0：功能与使用说明

[返回版本导航](README.zh.md) · [正式下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.6.0) · [该版本原始指南](https://github.com/Nothing1596/Piper-Policy/blob/robot-v0.6.0/docs/ROBOT-GUIDE.zh.md)

## 定位与安装

2026-09-27 发布的独立机器人中间件，资产 `piper-robot-0.6.0-win64-py312.zip`。要求 Windows x64 / Python 3.12，依赖由包内 wheels 离线安装。包含共享执行器、CLI、HTTP/MCP、MuJoCo 和机器人模型；不需要上位视频包。这里的下位机是电脑上的控制软件，不是固件。

核对发布页 `robot-SHA256SUMS.txt` 后解压，在包目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper-robot.cmd --help
```

安装创建 `.venv`，不加载模型、不连接 CAN；没有自动注册 PATH。可选旧安装补充包见 [installer-v1](installer-v1.zh.md)。0.6.0 不提供后来的单终端启动菜单、`/connect` 自动托管或 dev 系列观察计划功能。

## 仿真连接和看图

终端 A：

```powershell
.\piper-robot.cmd sim start --root work\sim --port 8798 --seed 200
```

保持运行，在终端 B：

```powershell
.\piper-robot.cmd --root work\sim --url http://127.0.0.1:8798 connect
.\piper-robot.cmd --root work\sim --url http://127.0.0.1:8798 status
.\piper-robot.cmd observe --root work\sim --url http://127.0.0.1:8798 --workspace . --output work\frame-01
```

frame-01 中包含真实仿真 RGB、depth 和标定。改变 seed、root 或端口时使用独立配置并同步各客户端 URL。不能仅看端口监听就宣称 ready。取图每次使用新路径；本版 observe 只支持 MuJoCo。

## 动作接口与模型使用

| 原语 | 参数含义 | 使用提醒 |
|---|---|---|
| `robot_move_joints` | 六个绝对关节角，度 | 不是增量角度 |
| `move_to` | 基座坐标 TCP 位置，米 | IK 端点运动不保证直线 |
| `move_by` | 指定 frame 下的相对位移 | 先核对当前坐标和姿态 |
| `move_linear` | 定姿态线性目标 | 不等于通用避障规划 |
| `set_gripper` | 绝对开度，米 | 力度是协议字段，不是已标定牛顿值 |

通过 `piper-robot ... <子命令> --help` 获取该版准确参数。动作返回 job_id 后，查询实际 job 状态并重新看图；不能把 accepted 当 succeeded。新动作有新 request_id，结果未知时先按原编号查询。共享执行器统一持有 CAN 和动作状态，模型不应直接再开设备。

外部代理使用包内 Python 的 `-m piperx_middleware.standalone_cli mcp`，配置 `--root`、`--url`、`--workspace` 的绝对路径。MCP 桥连接已运行服务，自己不启动仿真、不另占 CAN。工具链通常为 status→connect→simulation_observe→动作→查询 job→新图复核；模型策略仍是代理职责。

单独机器人包不需要 API key；若使用内置模型，模型服务要单独配置。0.6.1 修复了特定 DeepSeek 文本工具兼容，相关用户应先按该页升级。

## 停止、升级和排障

`stop` 请求软件停止，`shutdown` 在空闲时关闭服务并释放连接；它们不替代现场物理急停。401 常见于 token/root 与实例不匹配；connection refused 查终端 A、端口和进程；图像依赖/渲染错误与模型能力无关。

升级 0.7.1 请用新目录安装，沿其单终端流程操作，不把 0.6.0 的两终端管理方式当作新版要求。真机需厂商 SDK/驱动、明确设备身份、反馈与现场标定；安装本包不完成这些条件。

保存的独立包验证含新环境安装、MCP 握手、MuJoCo 连接、RGB-D、状态/stop 及关闭，未操作真实硬件。发布页下位源码回归为 48 通过/5 外部 SDK 跳过。本次没有重新运行这些测试。
