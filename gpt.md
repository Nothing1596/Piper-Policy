# Piper Policy 开发约定

## 单终端执行边界（2026-09-28）

- `piper-robot` 交互入口先明确选择 simulation/real，再选择 local/保存的 SSH target。无交互入口必须显式指定模式。
- 仿真、真机、不同远端分别保存配置、模型凭据和任务账本；旧配置备份迁移，不放宽只读、限位和锁存。
- `RobotService` 是 HTTP/MCP/CLI 共享的单一动作与 CAN 所有者。模型不能修改操作员策略；批准绑定具体请求与设备/配置/姿态。
- 真机默认 risk；auto 仅取消确认，不绕过硬限位、身份、反馈与故障检查。
- 控制会话 1 秒心跳/5 秒超时。失联完成已接受动作、拒绝后续、取消待审批；恢复查原 request_id，不自动重放。
- SSH 保留主机密钥验证，传结构化请求；远端须预装 SSH 与 CLI。执行器身份必须验证，不能仅按端口接管。
- ROS commissioning 独立；软件测试和仿真不能证明真实机械臂验收。不得自动发送真机动作。

## 协作与验证

按用户要求直接通过终端调用 agy/dsh/Kimi/Claude；不要为 CLI 再创建 Codex sub-agent。模块边界见 `docs/implementation/interaction-contract.md`。CLI 失败如实报告，不冒充其他模型结果。公共契约由集成负责人维护，模块作者各自提交测试。

运行测试：`python -m pytest piperx-cli/tests`。当前源包恢复了 `piper_aio` 固定版本的 MIT vendor 依赖；构建时保留许可证与来源。测试只使用隔离临时目录，禁止覆盖操作者真实配置。发行包必须附版本、哈希、操作说明及平台验证范围；没有明确授权不推送或公开发布。

## Windows 运行约定（2026-09-28）

进程存活检测必须区分平台：Windows 使用 OpenProcess/GetExitCodeProcess 并关闭句柄，禁止使用 `os.kill(pid, 0)` 作探测；其在 Windows 会终止目标。Python 句柄声明需兼容 64 位。未知/拒绝访问应保守避免重复启动。Windows 仿真复现脚本为 `tools/verify_windows_simulation.py`，只允许 MuJoCo 模式，使用独立 profile；不把仿真反馈计数当作物理 CAN 帧。
