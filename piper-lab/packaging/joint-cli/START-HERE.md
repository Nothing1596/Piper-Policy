# 从这里开始：Piper CLI 0.2.1

这是 **Windows x64 / Python 3.12** 的完整联合安装包。视频管线、模型 API 适配、MCP 工具和 PiperX 下位 CLI 装到一个环境；前端仍是命令行。

按这个顺序进行：

1. 将 ZIP 解压到固定目录，进入里面的 `piper-cli` 文件夹。
2. 准备 Python 3.12 x64，执行 `install.ps1`，再运行 `piper.cmd doctor`。
3. 先跑本地仿真，确认连接。此步不需要大模型。
4. 选择使用方式：配置模型 API 后运行 `policy run`，或者给 Codex / Claude Code 接入 MCP，让代理看图调用工具。
5. 需要参考视频时，再编译视频示范并带入任务。每次保留新的结果目录。

**完整的逐步说明、可复制命令、预期结果及故障处理： [新机器部署向导](docs/NEW-MACHINE-GUIDE.md)。**

包内已包含 0.2.1，不需要再找补丁 wheel。新增的 `simulation_observe` 能通过 MCP 返回当前仿真图像和保存 RGB-D 证据。

包内没有 Python 安装器、LM Studio、视觉模型权重、Codex/Claude Code 客户端、云端额度或真机驱动。依赖安装可离线完成；选择云端推理时才需要网络和你自己的模型账户。

不要把“安装完成”当作“模型已连通”，也不要把“动作已受理”当作“任务已完成”。向导在每一步都列出了验证办法。
