# RPent → PiperX 1.6 下载包验收（2026-11-01）

这是安装与自动化入口验收，模型任务成绩沿用单独保存的三题报告，不因发布包能启动而改记成功。

## 可下载包包含

- 固定版本的 adapter 源码、中文快速 README、详细参考、测试和三题模板。
- RPent `c9026c2bf9c1342b88cdc682e60d578770f871d3` 的 104 个锁定 Python 文件与许可证。
- 已发布、哈希不变的 Piper middleware 0.7.2 wheel。
- BUILD-INFO.json、逐文件 PACKAGE.sha256.json；发布附件另有 SHA256SUMS.txt 与 RELEASE-VALIDATION.json。
- 不包含凭据、执行器认证、实验运行目录或第三方视频/资产。首次安装和示例准备联网获取依赖及固定来源资源。

## 实际验证

| 检查 | macOS ARM64 | Featurize Linux |
|---|---|---|
| 从 ZIP 解压并独立安装 | 通过，目录含空格和 `&` | 通过，使用独立本地盘目录 |
| 锁定依赖与 RPent 源码检查 | 通过 | 通过 |
| 包内回归测试 | 130 passed | 130 passed |
| 双回合脚本化响应＋真实 MuJoCo | 2/2 通过 | 2/2 通过 |
| 合成 provider / 内部执行器请求 | 20 / 14 | 20 / 14 |
| 模型付费请求 | 0 | 0 |
| 示例资源与场景准备 | 上游下载 42 项并验 SHA256，三场景编译通过 | 42 项经哈希验证后复制到新缓存，再由 prepare 核验，三场景编译通过 |
| 三题 dry-run | 3 题，blocked=[] | 3 题，blocked=[] |
| 清理 | stopped/drained/shutdown 均确认；专用 PID 已退出 | 同左 |

开发侧另有 150 项检查通过，其中包含 20 项原 RPent 兼容检查；这 20 项外部参考测试不宣称随包分发。直接运行包内测试需带 `RPENT_ROOT` 和 `PYTHONPATH`；quickstart 已自动为自己的子进程设置它们。

## 保留的失败与边界

- macOS 最早一次自测在新环境初始化阶段出现 stale feedback，执行器返回 outcome_unknown 并暂停；清理已确认，没有重放动作。另开独立自测输出后两题通过，没有放宽反馈阈值。原记录保留，不能声称所有冷启动都无此问题。
- 最早一次直接 pytest 未设置 RPent 导入路径，4 个测试导入失败；使用下方明确环境后 130 项通过，不算产品模型失败。
- 先前模型三题仍为两题独立失败、一题基础设施异常；180° 附近数值限位拒绝和持物保持问题没有在打包阶段修复或隐藏。
- 没有新增付费模型回合、原生 Windows、真实 CAN、相机或实机验收。Windows 使用 WSL2，但没有单独验收 WSL2。

包内测试命令（在解压目录）：

```sh
RPENT_ROOT="$PWD/deps/RPent" PYTHONPATH=".:$PWD/deps/RPent" .venv/bin/python -m pytest tests -q
```

发布前以新包和已验收候选包的逐文件比较，确认启动器、运行源码、锁定依赖、RPent 和示例模板不变；最终差异只允许验证文档、README 和构建身份。发布资产哈希再次与本机文件核对。保留 Robot 0.7.2 的 Latest 状态，本组件是独立实验版。
