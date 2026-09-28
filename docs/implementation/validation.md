# Piper Robot 0.7.0 验证记录

日期：2026-09-28。基线 `639ba59`，集成分支 `codex/single-terminal`。本轮只做软件实现、测试与本地发行包；没有发布、推送或真实 CAN 写入。

## CLI 分工与实际调用

| 板块 | 实际入口 | 交付与复核 |
|---|---|---|
| 前端 | `agy --model gemini-3.8-flash-high` | 模式/命令/确认/审批/退出；Codex 应用结构化输出并复核；慢握手心跳修订也由同一 CLI 提供 |
| 执行器/SSH/配置 | `dsh --profile headless`，现有配置模型 | 进程、会话、隧道、身份与迁移；按集成反馈修复限制迁移、远端退出误报和旧隧道残留 |
| 审批/绑定 | `kimi -p`，现有配置模型 | 三种审批模式、有效限位、逐点位移检查、批准绑定及测试 |
| 独立审查 | `claude -p --model claude-opus-5-5` | 4 项独立回归通过；报告附集成复核限定，不能替代全套验收 |
| 共享接口/集成 | Codex | HTTP/MCP/持久化/前端运行包装、进程与故障测试、打包 |

最终协作通过实际终端 CLI 完成，没有用 Codex sub-agent 冒充这些 CLI。Claude 的 `opus5-5` 调用曾被供应商拒绝；核对配置后用完整 ID `claude-opus-5-5` 成功响应并完成审查。发行包不包含供应商凭据或 CLI 私有日志。

## 本地全量测试

macOS arm64，Python 3.12.14：**494 passed，1 warning，34.68 秒**。警告为 Starlette TestClient 的 httpx 弃用提示，不是测试失败。

```sh
PIPERX_SDK_ROOT=/path/to/vendor-sdk python -m pytest -q piperx-cli/tests
```

其中 5 项 SDK 序列化测试使用本机已有的固定 SDK 与内存 CAN 替身，没有打开物理总线。发行包未附厂商 SDK；缺少 SDK 时这些测试按既有规则跳过。其余测试使用临时隔离目录。

| 验证内容 | 证据/范围 |
|---|---|
| 模式、配置与凭据隔离；旧只读/限位/锁存迁移 | `test_profiles.py`，旧数据字节备份、严格校验及独立凭据 |
| always/risk/auto 和边界 | `test_approval_policy.py`、`test_approval_binding.py` |
| HTTP/MCP 同一规则、模型不能写操作员配置 | `test_interaction_integration.py`，真实协议请求/测试服务 |
| 姿态、配置、停止、连接变化使批准失效 | 同上及独立审查测试 |
| 新进程、共享复用、正常/异常退出 | `test_managed_console_process.py`，本机真实子进程 |
| 5 秒失联后完成原动作并拒绝新动作 | 会话假时钟与执行器集成测试；本机前端终止测试使用实际超时 |
| 启动慢握手 | 5.2 秒延迟超过租约，心跳持续；修复前失败、修复后通过 |
| 已接受动作的响应丢失 | 本机真实 HTTP 代理先转发动作再丢弃响应，按原 ID 查询完成结果，没有重发 |
| 执行器重启/身份变化、未知结果、幂等 | `test_runtime.py`、`test_managed_runtime.py` 与集成测试 |
| SSH 断网/忙碌/未确认/错误身份响应及隧道清理 | 故障替身测试；未声称实际跨主机 SSH 验收 |

## 打包与干净安装

0.7.0 wheel 在独立 Python 虚拟环境中安装，`uv pip check` 通过，48 个依赖相容。离开源码目录验证 `piper-robot --help`、真实 MCP 握手、sim/MuJoCo 连接、模拟关节动作、按原请求编号查询及退出后 runtime 记录移除。中文与空格目录通过。

最终压缩包包含源码、测试、MuJoCo 资源、wheel、安装入口、许可证、中文指南和文件 SHA-256 清单。首次安装需要网络下载平台依赖；**这不是内置 Python 的离线便携包**，也不含厂商 CAN 驱动或 SDK。包内不含用户配置、令牌、数据库和模型 API key。

构建命令：

```sh
uv build piperx-cli --wheel --out-dir dist/interactive-wheels
python tools/build_robot_interactive.py --wheel dist/interactive-wheels/piperx_middleware-0.7.0-py3-none-any.whl --output dist/interactive-release
```

## 未覆盖平台与后续验收

- Windows 11 Parallels 当前处于 suspended；本轮未启动它或替换其服务，Windows/WOA 包安装、设备驱动和进程行为仍需现场验证。
- 可用 Linux SSH 目标本次连接被关闭，未完成 Linux 进程或跨主机 SSH 实测。
- 没有机械臂、CAN 占用竞争、实时反馈、相机标定、碰撞规划、急停或抽屉任务的实物验收。
- ROS commissioning 独立且保持原值，软件测试不能代替硬件确权。
- 独立审查只覆盖其报告所列测试；修复与全量集成证据以上表为准，不把模型的笼统结论当作证明。

下一步是从压缩包在 Windows/Linux 各完成仿真安装与远程故障实测，再单独安排真机连接和运动验收。

## 后续 Windows 实测更新

用户随后要求在 Windows 主机运行仿真，已完成。结果及发现的 Windows 进程探测修复见 [Windows 仿真记录](windows-simulation.md)。本文件前面的 494 项结果和原压缩包校验只对应修复前的 macOS 阶段，不能覆盖或替代该后续实测。

## 推送前最终回归（2026-09-28）

包含 Windows 进程探测修复的当前代码：macOS **508 passed、1 Windows 专用 skipped、1 弃用 warning，34.54 秒**。首次回归发现停止测试在读取 actuator hold target 前允许仿真推进一步，造成时序失败；将停止前后比较放到同一 RLock 内，保留原有误差断言，全量通过，且该测试独立重复 5 次均通过。只调整测试同步方式，没有改变停止实现或放宽容差。Windows 实测仍以上述独立记录为准。
