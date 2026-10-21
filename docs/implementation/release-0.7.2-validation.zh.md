# Robot 0.7.2 / Video 0.3.1 发布验证（2026-10-21）

本轮经用户明确授权分批提交、推送并发布。两个组件使用同一源码提交；最终 commit 和 wheel SHA-256 写入交互包 BUILD-INFO.json，制品和包内文件另有校验清单。

## 已执行的软件检查

- 上下文、session、plan 和反馈聚焦检查：41 项通过。
- 机器人、视频和实验工具的首次完整检查：1392 passed / 12 skipped，117.31 秒；随后增加三个打包回归检查，最终完整检查为 **1395 passed / 12 skipped**，117.75 秒；原始 JUnit/日志保留在本地发布目录，发布资产另保存摘要。
- 新打包检查与 token_context 聚焦检查：15 项通过，验证完整版本文档随包、组件来源与 hash、Python 要求、敏感运行文件排除，以及版本不符时拒绝构建。这 15 项和全量重叠，不能相加。
- Claude CLI 使用提供的上下文代码/测试进行无工具只读审查，回执返回配置模型别名 claude-opus-5-5，未提出发布阻塞；代码复核与实际 pytest 由 Codex 完成，不把 CLI 返回别名当供应商权重证明。

12 个跳过分别为：Windows 原生进程探测 1 项、未配置外部 SDK 的协议检查 7 项、未安装 LeRobot 的导出 1 项、ROS 环境 catkin_pkg 2 项、显式真实 CLI fixture 1 项。跳过不计通过。既有 Starlette/httpx 弃用警告单列，未为本次发布升级产品依赖。

## 构建和安装核对

最终发布从已提交源码的 Git archive 构建，不从混有工作文件的目录直接打包。两个 wheel 的元数据、代码/资源与源码比对；交互 ZIP 包含完整 docs 树，安装器检查文件和版本，附独立视频 wheel。模型权重、凭据、现场 profiles、私有研究 GT 和独立 seed 交付均不纳入。

构建使用独立临时 Python 环境；不改用户已有安装。最终隔离安装、CLI/视频解码和有限仿真冒烟的逐项结果，以及下载资产的核验，以发布页附件为准：

[RELEASE-VALIDATION.json](https://github.com/Nothing1596/Piper-Policy/releases/download/robot-v0.7.2/RELEASE-VALIDATION.json)

这份附件在制品构建后生成，避免为了把 ZIP hash 写回源码而改变构建输入。SHA256SUMS.txt、ZIP 自身 .sha256 和 BUILD-INFO.json 各自标明所校验的对象，不混作同一个身份。

## 证据范围

本机为 macOS ARM64；本轮没有重新运行 Windows/Linux 新安装、真实 CAN、D435/相机标定或真实机械臂任务。有限仿真冒烟验证安装链和工具/图像/plan/退出，不是新的完整 LongVIL 成功率或模型性能实验。没有付费模型机器人任务，没有重跑或改写旧评分结果。

token_context 是发布源码中的实验模块，未自动接入日常模型循环。独立 seed 接口仍停在其既有隔离检查点，不作为本版 grounding 或跟踪成绩。旧 release 资产及溯源分支保持原样。
