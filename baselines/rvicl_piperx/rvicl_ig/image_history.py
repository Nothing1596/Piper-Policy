"""Request-time image selection; robot text and demonstration images stay available."""

import json
from dataclasses import replace

DEMO_TOOLS = {"view_task_video", "view_demo_clip"}
ACTION_TOOLS = {"move_to", "rotate_wrist", "set_gripper", "release", "move_both", "wait"}
LIVE_TOOLS = {"view_env_state", *ACTION_TOOLS}
OMITTED_IMAGE = "[earlier execution image omitted; text history retained]"
LABEL_PREFIXES = ("Current observation image:", "Historical observation image:",
                  "Previous primitive execution process image:", "Historical primitive execution process image:",
                  "Human demonstration image:", "Tool result image:")


def image_caption(item, *, historical=False):
    kind, camera = item["image_kind"], item.get("camera", "unknown")
    source = (f"camera={camera}; simulation_step={item.get('simulation_step', 'unknown')}; "
              f"source_tool={item.get('source_tool', 'unknown')}; "
              f"source_turn={item.get('source_turn')}; source_call_id={item.get('source_call_id')}")
    if kind == "execution_process":
        prefix = "Historical" if historical else "Previous"
        label = (f"{prefix} primitive execution process image: frame {item['sequence_order']} of "
                 f"{item['sample_count']} (chronological order); phase={item['phase']}; {source}. "
                 "This is a recorded execution frame, not the current scene.")
        if camera in {"left_wrist", "right_wrist"}:
            label += f" Wrist belongs to tool arm=\"{camera.removesuffix('_wrist')}\"."
        return label
    if kind == "current_observation":
        prefix = "Historical" if historical else "Current"
        if item.get("camera_group"):
            return (f"{prefix} observation image: {camera}; {source}; "
                    f"capture_stamp_s={item.get('source_stamp_s')}; sim_time_s={item.get('sim_time_s')}; "
                    f"camera group {item['camera_group']} captured sequentially, not hardware synchronized.")
        return f"{prefix} observation image: {camera}; {source}; synchronized head/left_wrist/right_wrist observation."
    if kind == "demonstration":
        return "Human demonstration image: " + item.get("description", item.get("caption", ""))
    return "Tool result image: " + item.get("description", item.get("caption", "")) + "; " + source


def describe_tool_images(result, tool, *, turn=None, call_id=None):
    """Attach machine-readable provenance and an English caption to every image."""
    images = []
    for index, original in enumerate(result.get("_images", [])):
        item = dict(original)
        item.setdefault("source_tool", tool)
        item.setdefault("source_turn", turn)
        item.setdefault("source_call_id", call_id)
        if "image_kind" not in item:
            item["image_kind"] = ("demonstration" if tool in DEMO_TOOLS
                                  else "current_observation" if tool in LIVE_TOOLS else "tool_result")
        if item["image_kind"] == "current_observation":
            item.setdefault("camera", ("head", "left_wrist", "right_wrist")[index % 3])
            item.setdefault("simulation_step", result.get("state", {}).get("steps"))
        item.setdefault("description", item.get("caption", ""))
        item["caption"] = image_caption(item)
        images.append(item)
    return {**result, "_images": images} if images else result


def latest_execution_images(messages):
    """Keep the latest action process, latest live three views and tool overlay."""
    return _image_history(messages, latest_only=True)


def label_image_history(messages):
    """Place captions immediately before image bytes, including budget mode."""
    return _image_history(messages, latest_only=False)


def _image_history(messages, *, latest_only):
    """Restore captions after RPent separates tool text from binary image parts.

    Per-image provenance stays on BinaryContent between requests so a partial
    trim (e.g. replacing three views while retaining the last action process)
    cannot shift the association between surviving bytes and their captions.
    """
    from pydantic_ai import BinaryContent
    from pydantic_ai.messages import ToolReturnPart, UserPromptPart

    def is_image(item):
        return isinstance(item, BinaryContent) and item.media_type.startswith("image/")

    artifacts, groups = {}, []
    for mi, message in enumerate(messages):
        sources = []
        for part in message.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            data = part.content
            if isinstance(data, str):
                try:
                    data, _ = json.JSONDecoder().raw_decode(data.lstrip())
                except ValueError:
                    continue
            if not isinstance(data, dict):
                continue
            images = data.get("planner_images")
            if images:
                described = describe_tool_images({"_images": images, "state": data.get("state", {})}, part.tool_name)["_images"]
                for item in described:
                    artifacts[data["planner_image_step"], item["name"]] = item
                sources.append((part.tool_name, described))
            elif part.tool_name == "read_image" and "artifact" in data and "step" in data:
                original = artifacts.get((data["step"], data["artifact"]), {})
                item = {**original, "image_kind": "demonstration" if original.get("image_kind") == "demonstration" else "tool_result",
                        "source_tool": "read_image", "description": f"Recorded artifact {data['artifact']}, planner step {data['step']}"}
                sources.append(("read_image", [item]))
        prompts = [(pi, part) for pi, part in enumerate(message.parts)
                   if isinstance(part, UserPromptPart) and isinstance(part.content, list)
                   and any(is_image(item) for item in part.content)]
        for index, (pi, part) in enumerate(prompts):
            tool, metadata = sources[index] if index < len(sources) else ("unknown", [])
            items = []
            for ii, item in enumerate(value for value in part.content if is_image(value)):
                info = (item.vendor_metadata or {}).get("rvicl_image")
                if info is None:
                    info = metadata[ii] if ii < len(metadata) else {"image_kind": "tool_result", "source_tool": tool}
                items.append((item, info))
            groups.append({"key": (mi, pi), "tool": tool, "items": items})

    latest_live = next((g["key"] for g in reversed(groups)
                        if any(info["image_kind"] == "current_observation" for _, info in g["items"])), None)
    latest_action = next((g["key"] for g in reversed(groups) if g["tool"] in ACTION_TOOLS), None)
    latest_extra = next((g["key"] for g in reversed(groups)
                         if any(info["image_kind"] == "tool_result" for _, info in g["items"])
                         and (latest_live is None or g["key"] > latest_live)), None)
    selected = list(messages)
    for group in groups:
        mi, pi = group["key"]
        message = selected[mi]
        content, images = [], iter(group["items"])
        for value in message.parts[pi].content:
            if isinstance(value, str) and value.startswith(LABEL_PREFIXES):
                continue  # Regenerate our captions; never duplicate them on retries.
            if not is_image(value):
                content.append(value)
                continue
            value, info = next(images)
            kind = info["image_kind"]
            keep = (not latest_only or kind == "demonstration"
                    or kind == "current_observation" and group["key"] == latest_live
                    or kind == "execution_process" and group["key"] == latest_action
                    or kind == "tool_result" and group["key"] == latest_extra)
            if not keep:
                content.append(OMITTED_IMAGE)
                continue
            historical = (kind == "current_observation" and group["key"] != latest_live
                          or kind == "execution_process" and group["key"] != latest_action)
            content.extend([image_caption(info, historical=historical),
                            replace(value, vendor_metadata={**(value.vendor_metadata or {}), "rvicl_image": info})])
        parts = list(message.parts)
        parts[pi] = replace(parts[pi], content=content)
        selected[mi] = replace(message, parts=parts)
    return selected
