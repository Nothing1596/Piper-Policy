# 0.7.2.dev2 开发包

连续视频、示范管线接入和分层观看的本地开发包，尚未公开发布。

- `/video`、`/observe` 复用 RGB-D 相机进程，保存时序证据与状态对齐信息。
- `/demo watch` 支持地址、直接/管线模式、时间点/区间，管线模式返回 CV/YOLO 及可选低思考视觉 API 的评价。
- `/demo prepare|compile|use|show|clear` 连接原 piper-video；示范方法与实时场景分别交给执行模型。
- `--agent` JSONL 和 `--agent-mcp` stdio MCP 使用同一前端与执行器，沿用审批、限位及稳定 request_id。
- 联合包保持两个独立依赖环境，并附 Mac .command 启动器。相机与真实动作的验收范围分别记录。
