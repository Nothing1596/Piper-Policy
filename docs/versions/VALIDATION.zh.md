# 版本说明核验记录（2026-10-21）

[返回版本导航](README.zh.md)。本页记录版本资料与使用文档检查，和随后用户授权的分批推送/发布准备区分；没有修改执行器、运动、模型循环、物理参数或研究评分。

## 资料核对

- 通过 GitHub API 核对远端 main、所有公开 release 的 tag/资产及分支身份；与本地 Git 对照。结构化快照保存在 `status-20261021.json`。
- 六个正式发布项分别参考自己的 tag、发布说明和原始安装指南。dev1–dev4 没有正式 tag/release，明确注明为开发阶段，不虚构同名下载包。
- 对照当前 pyproject、CLI 参数、手动解析器和工具 schema；主源码与本地未提交模块分开列示。
- 对照原冻结 seed 交付说明，但没有复制研究图片、GT、映射或整套研究产物进入仓库，也没有运行研究回放。

## 本轮实际执行的检查

现有 macOS Python 环境中，以下四个命令帮助入口都退出 0：

```text
python -m piperx_middleware.standalone_cli --help
python -m piperx_middleware.standalone_cli observe --help
python -m piperx_middleware.standalone_cli mcp --help
python -m piperlab.video_entry --help
```

这四项只查看参数，不启动执行器、模型、MCP 服务、相机或仿真。校验器导入真实手动解析器及工具 schema，底层 client 使用禁止任何 I/O 的替身，仅列工具定义；对文档中的完整 manual、watch、observe 和 JSONL 示例做语法/schema 检查。TASK_ID/VERSION 等明确占位值仅在检查内替换为合法示例，动作 request_id 按人工前端的补齐约定处理；没有派发工具。

另外检查新增文档的本地链接、六个发布 tag 的页面身份、源码路径和 `git diff --check`。初次链接检查发现本核验页尚未创建，补齐后重新核对；没有为通过文档检查修改产品代码。最终计数保存在 `validation-20261021.json`。

## 没有验证的内容

没有新建联网安装环境、重新构建或上传包，没有运行完整 pytest、付费模型任务、仿真动作、Windows/Linux 安装或真实硬件。版本页中的历史测试和实验数字均为既有保存记录，不代表本轮重跑，不把不同轮次相加。

文档检查结束时没有自动推送或发布。随后用户明确授权，token_context 实现/测试及说明先由 956f916 提交并推送，版本说明作为下一批提交；正式发布验证另列。溯源分支和独立冻结研究交付未被改写，旧下载资产保持原样。
