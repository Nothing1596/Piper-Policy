# 下位机管线独立部署指南

[English](ROBOT-GUIDE.en.md)

[完整命令与功能参考](ROBOT-CLI-REFERENCE.md)

版本 0.6.0，入口 `piper-robot`，原 `piperx` / `piperx-mcp` 仍可用。此包独立安装共享执行器、CLI、HTTP/MCP、MuJoCo 和机械臂模型，不安装视频解析包、检测器或视觉大模型。这里的“下位机”指主机上的控制中间件，不是机械臂固件。

## 1. 安装

准备 Windows x64 和 Python 3.12 x64。解压到新目录，在目录中运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper-robot.cmd --help
```

安装器核对 SHA256，离线安装到 `.venv`，不修改全局 PATH。无 `py` 启动器时传 `-PythonExe C:\Python312\python.exe`。新机器重新安装，不复制已有 `.venv`。安装本身不会接通或使能真实机械臂。

## 2. 启动仿真并连接

终端 A，保持运行：

```powershell
.\piper-robot.cmd sim start --root work\sim --port 8798 --seed 200
```

终端 B：

```powershell
.\piper-robot.cmd --root work\sim --url http://127.0.0.1:8798 connect
.\piper-robot.cmd --root work\sim --url http://127.0.0.1:8798 status
.\piper-robot.cmd observe --root work\sim --url http://127.0.0.1:8798 --workspace . --output work\frame-01
```

`frame-01` 包含 RGB、深度和标定信息。每次使用新输出目录，防止覆盖证据。`observe` 当前仅支持 MuJoCo 相机。启动器检查已有 root 的后端、端口和种子；改变参数请使用新 root。端口已占用时换端口，所有命令需同步修改。

## 3. 接入模型或代理

此包本身不需要配置模型。Codex、Claude Code 或其他代理负责理解用户指令和看图，通过 MCP/终端调用工具。std​io 配置示例：

```json
{
  "mcpServers": {
    "piper-robot": {
      "command": "D:/PiperRobot/.venv/Scripts/python.exe",
      "args": ["-m", "piperx_middleware.standalone_cli", "mcp", "--root", "D:/PiperRobot/work/sim", "--url", "http://127.0.0.1:8798", "--workspace", "D:/PiperRobot"]
    }
  }
}
```

替换绝对路径，并按客户端的 MCP 设置填写命令和参数。先启动执行器；MCP 桥只连接它，不另开 CAN。token 从 root 的本地文件读取，不要把 token 放进提示词或仓库。省略 `--workspace` 时不暴露取图工具。

代理流程：调用 `robot_status` → 必要时 `robot_connect` → `simulation_observe` 看当前图像 → 用 `move_to`、`move_by`、`move_linear`、`set_gripper` 等工具动作 → 查询返回的 job_id → 再取图复核。相同动作重试沿用 request_id，新的动作使用新 ID。调用返回 job_id 只代表已提交，必须检查最终完成状态和画面。

终端路径也可用：`piper-robot --root ... --url ... <命令>`。各子命令参数用 `--help` 查询。停止用 `stop`，关闭服务用 `shutdown`；不要把进程退出等同于物理急停。

## 4. 结构

```text
外部模型/代理
   ├─ piper-robot CLI
   └─ MCP 工具 → HTTP 客户端
                 ↓
       单一共享执行器：认证/请求去重/队列/反馈/限位
                 ↓
       MuJoCo 仿真 或 配置后的真实 CAN 后端
                 ↓
             状态 / job / RGB-D 证据
```

执行器统一持有连接和动作状态。模型不直接占用 CAN。动作前需要校验当前反馈、坐标和范围；抓手接触、抬起成功和最终摆放成功是不同条件。

视频示范管线另外安装 `piper-video`。一个代理可同时连接两边 MCP：先读示范，再看当前机器人画面执行。视频解析结果不会自动转换成硬件指令，也不替代实时闭环。原联合包 v0.2.1 仍提供旧的一体化策略入口。

## 5. 真机与验收范围

先完成上述仿真。真实 CAN 后端需要对应平台、驱动、厂商 SDK、正确机械臂型号以及现场标定；这些不因离线包安装成功而具备。Windows 包包含 `python-can`，不包含厂商 SDK，也未验证 Windows 上真实 CAN 执行。用 `piper-robot init --help` 查看初始化选项，正式使能前按设备现场流程核对急停与工作空间。

本版本的独立包验证覆盖离线安装、MCP 握手、仿真连接和真实 RGB-D 回传；历史彩块任务结果不扩展为真机验收。包中 `licenses` 保留机械臂模型等第三方许可；原创代码尚未指定统一许可证。
