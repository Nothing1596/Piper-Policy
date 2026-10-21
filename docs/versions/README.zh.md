# Piper 各版本功能与使用指南

核对日期：2026-10-21。这里区分正式下载包、源码开发阶段和未合并研究代码；不是新的发布公告。

## 推送检查与后续批次

首次核对时，GitHub `main` 与本地 HEAD 同为 [`09256a50935bfc237f7dd737b718dffe22e3d5e0`](https://github.com/Nothing1596/Piper-Policy/commit/09256a50935bfc237f7dd737b718dffe22e3d5e0)。这部分已经推送。GitHub Latest 仍是 [robot-v0.7.1](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.7.1)，没有 0.7.2 正式 release。

随后经用户授权分批推送：第一批 [`956f916`](https://github.com/Nothing1596/Piper-Policy/commit/956f916) 已纳入并推送 `tools/token_context.py`、对应测试与说明；相关 41 项检查通过。它仍是实验适配器模块，没有自动启用到日常 CLI。这里的首次检查快照保留在 `status-20261021.json`，不是持续更新的远端状态。独立首帧 seed 研究目录仍不在本仓库中。

另外，远端 `experiment/derivation` 仍在 `9f12a69`，只整理到失败收尾阶段，**目前并未与 main 的实现保持一致**。它缺少后续行为复核、部分几何/重连和实验修订；本轮未擅自重写这条溯源历史。

## 选择一个入口

| 版本或阶段 | 主要用途 | 入口与安装形态 | 详细说明 |
|---|---|---|---|
| 联合版 v0.2.1 | 复现历史视频→视觉策略→仿真流程 | Windows x64 / Python 3.12 离线包，`piper.cmd` | [功能、安装与任务流程](combined-0.2.1.zh.md) |
| Video 0.3.0 | 单独处理示范视频、模型解析和证据 | Windows 离线包，`piper-video` | [功能与使用](video-0.3.0.zh.md) |
| Robot 0.6.0 | 单独提供执行器、机器人 CLI、HTTP/MCP | Windows 离线包，多终端方式 | [功能与使用](robot-0.6.0.zh.md) |
| installer-v1 | 给 Video 0.3.0 / Robot 0.6.x 补安装及 PATH 注册 | 小型脚本附件，不是完整程序 | [安装及撤销](installer-v1.zh.md) |
| Robot 0.6.1 | 已安装 0.6.0 的 DeepSeek 工具协议修正 | 更新 wheel，不含完整依赖 | [升级及模型配置](robot-0.6.1.zh.md) |
| Robot 0.7.1 | 稳定的单终端控制 | 联网安装交互包，`piper-robot` | [完整操作说明](robot-0.7.1.zh.md) |
| Robot 0.7.2.dev1 | 持续相机、选帧、JSONL 模型入口 | 未发布开发阶段 | [功能及适用范围](robot-0.7.2.dev1.zh.md) |
| Robot 0.7.2.dev2 | 示范观看、视频管线与前端 MCP | 未发布开发阶段 | [示范与实时场景工作流](robot-0.7.2.dev2.zh.md) |
| Robot 0.7.2.dev3 | 持久 plan、观察 skill、可复用几何计算 | 未发布开发阶段 | [计划与计算使用方法](robot-0.7.2.dev3.zh.md) |
| Robot 0.7.2.dev4 | 打通 MuJoCo 观察→plan→几何闭环 | 未发布开发阶段 | [仿真模型闭环](robot-0.7.2.dev4.zh.md) |
| main@09256a5 | 后续反馈、行为复核、实验与评分修订 | 已推送源码；机器人包版本仍为 0.7.2.dev4 | [源码安装与逐批变化](source-09256a5.zh.md) |
| experiment/derivation@9f12a69 | 查阅迭代原因和代码演绎 | 已推送溯源分支，非完整运行发行包，落后 main | [阅读与差异说明](experiment-derivation.zh.md) |
| 原本地工作与独立研究 | 已提交的 MoE 上下文修复、尚独立保存的 seed 接口 | 实验模块与研究交付分别说明 | [当前状态及使用边界](unmerged-work-20261021.zh.md) |

六个正式发布项的 tag 都存在；dev1–dev4 没有对应公开 release/tag。开发阶段文档用于说明演进和现有能力，不能据此假定存在可下载的同名完整包。后续源码一直沿用 dev4 元数据，因此还必须记录 commit；不能仅凭版本字符串判断是否有最新 harness。

## 功能如何逐步增加

| 能力 | 首次对应阶段 | 实际作用与边界 |
|---|---|---|
| 视频选帧、阶段与证据包 | 联合版；Video 独立版保留 | 提供示范证据，不训练模型权重 |
| 共享机器人执行器、运动原语、MCP | Robot 独立版 | 主机控制中间件，不是机械臂固件 |
| 模式选择、自动连接、同终端审批 | Robot 0.7.1 | 不再要求另一个终端 OPEN；执行约束仍有效 |
| 连续相机与变化选帧 | dev1 | 帧变化不等于接触、抓取或插入成功 |
| `/demo watch` direct / pipeline | dev2 | 区分录制示范与当前场景；评价可被原图反驳 |
| `task_plan` / `scene_query` / skill | dev3 | plan 保存解释和问题，几何后端提供可引用测量 |
| MuJoCo 同接口看图与测量 | dev4 | 模拟标定与真实相机配置分开保存 |
| 软 plan、受阻恢复、持久证据、行为提醒 | 后续 main | 提供反馈和复核，不替模型判断任务成功 |
| token 工作集与确定性公开回执压缩 | 源码 956f916 | 已推送实验模块，日常 CLI 未自动启用 |

## 阅读和升级规则

1. 首次使用单终端控制，读 0.7.1；需要视频、plan 或模型仿真闭环，读源码版并明确使用开发代码。
2. 只处理视频，选 Video 0.3.0；无需为选帧安装机器人执行器。
3. 只有旧环境才沿用 0.6.x 或联合版。不要把旧 `installer-v1` 覆盖到 0.7.1/开发包。
4. 使用新目录和独立环境升级；保留旧配置和结果，不复制 `.venv`，不覆盖运行中的执行器。
5. 文中的示例优先使用仿真。安装、协议测试、模型任务与真机验收是不同证据；历史测试数都标明为保存记录，本次没有重跑硬件或模型实验。

详细公共参考：[机器人命令](../ROBOT-CLI-REFERENCE.md)、[视频命令](../VIDEO-CLI-REFERENCE.md)、[单次任务演示](../ONE-TASK-DEMO.zh.md)。本轮文档核验见 [VALIDATION.zh.md](VALIDATION.zh.md)。
