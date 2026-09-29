# Piper Robot 0.7.1 — 单终端交互

合并单终端重构与交互简化。视频包仍为 0.3.0，旧联合包及历史安装器保持独立。

## 用户能直接使用的变化

- `piper-robot` 启动先选择仿真/真机、本机/已保存远端；`/connect` 自动管理执行器、端口和设备选择。
- `/manual`、`/model`、`/status`、`/tools` 与 `/approval`、`/limits`、`/config`、`/remote`、`/mode` 在同一终端完成操作，不再要求另开终端开放时间窗口。
- 新仿真默认自动执行合法动作；真机默认风险动作确认。模型不能修改操作员限制，全自动也保留限位、反馈和故障检查。
- 单个配置提案直接 `/confirm`；限位向导只询问当前审批模式需要的字段。手动调用不需要模型 API key。
- 执行器统一持有 CAN；会话失联不重发未知动作，退出清理本次专用执行器。共享服务只释放当前会话。Windows 进程检查使用不会终止目标的 API。
- 保留原工具名、单位、请求编号和脚本入口；旧配置备份迁移并保留限制。旧客户端仍须满足控制会话要求。

## 下载与使用

下载 `piper-robot-0.7.1-interactive.zip` 及 `.zip.sha256`。ZIP 包含 wheel、源码、测试、MuJoCo 资源、安装入口、中英文指南、单次任务演示及文件校验清单。单独 wheel 供已有 Python 环境使用。

先准备 Python 3.11+（验证使用 3.12），解压到新目录。Windows 执行 `py -3 install.py` 或 `Setup.cmd`，随后 `.\piper-robot.cmd`；macOS/Linux 执行 `python3 install.py`，随后 `./piper-robot`。首次安装联网下载依赖，不自动安装 Python 或注册 PATH，不包含厂商 CAN 驱动/SDK。升级前退出旧程序，不覆盖运行中的包或复制 `.venv`。

[中文指南](ROBOT-GUIDE.zh.md) · [English guide](ROBOT-GUIDE.en.md) · [单次任务演示](ONE-TASK-DEMO.zh.md)

## 验证及已知限制

- macOS 全量：**533 passed、6 skipped**（Windows 专用探测 1 项、未提供 SDK 的协议测试 5 项）；另有一项既有依赖弃用警告。
- 全新 wheel 安装与依赖检查通过；安装包的 MCP/HTTP 仿真完成 J1 5°、夹爪 40 mm、关节归零，退出清理通过；另有三次独立冷启动通过。
- 首轮安装冒烟有一次 Ready 组合断言失败，未保存当时状态，发生于任何动作前。后续加日志的一次及三次冷启动未复现，尚不能确定原因。
- Gemini 3.8 Flash、dsh 静态审查通过；Kimi 聚焦校验器与演示签名审查通过。Claude 本轮调用超时，不计入通过。
- 本版未重做 Windows/Linux 和跨主机 SSH 实机验收；历史 Windows 仿真记录有单独范围。未控制真实机械臂，未完成相机标定、碰撞规划或真实任务成功率评测。

详细证据见 [验证记录](implementation/validation.md) 和 [多 CLI 审查](implementation/peer-review-0.7.1.md)。
