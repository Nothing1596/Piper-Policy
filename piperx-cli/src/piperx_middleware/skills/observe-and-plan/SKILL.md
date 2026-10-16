---
name: observe-and-plan
description: Resolve manipulation unknowns from demonstration and current evidence, then maintain a concise plan when persistence is available.
---

# Observe and plan

## Shared observation guidance

Use this observe-and-plan workflow to resolve questions that change an action decision. It is not another motion permission gate.

1. Read the supplied initial scene and chronological demonstration images. Reconstruct candidate actions and the visible before/after object states. The initial configuration alone does not establish an action occurred.
2. Identify a few consequential unknowns: did the manipulation occur, which object moved, what was its reference object/final relation, and which action happened first? Do not make a checklist of irrelevant details.
3. Inspect the relevant time interval or exact frame with demo_watch when the existing evidence cannot distinguish alternatives. Verify temporal changes and the final spatial relation separately. Distinguish a hand adjustment from an object transfer; do not omit a visible deliberate transfer merely because it seems incidental.
4. Use original images ahead of detector labels and fast-vision proposals. Reject a hypothesis contradicted by visible evidence. A valid depth, detector confidence, or model summary does not establish a grasp or task success.
5. Revise only the unsupported interpretation. Cite supplied source/frame IDs and timestamps in a concise explanation; retain genuine ambiguity rather than fabricate evidence. Keep the demonstrated order and infer only actions supported by the clip. Do not expose or narrate a long private reasoning trace.

## Offline demonstration context

This is recorded-video interpretation. The first real decoded frame is the RECORDED INITIAL SCENE and demonstration start, not a live camera. Compare it with subsequent recorded frames. There is no robot feedback, calibration or depth measurement unless explicitly supplied; do not request unavailable live/geometry tools or invent metric coordinates. Source for recorded questions is demo; omit query for semantic questions.

## Plan persistence

Use task_plan for persistence. If the frontend supplies an existing task, use its returned task_id and version. Otherwise create a task with the user's goal. Never substitute a session/video-path UUID for a task_id. Persist only consequential questions still needing evidence. Put visual conclusions with source/frame citations in changes.interpretation; remove resolved demo questions from the pending list. Retain measured geometry questions and their backend-owned results. Unknown questions may remain pending and must be reflected in the final uncertainties.

Correct tool shapes (substitute values returned by tools):
- Create: {"operation":"create","goal":"USER_GOAL"}
- Read: {"operation":"get","task_id":"RETURNED_TASK_ID"}
- Update: {"operation":"update","task_id":"RETURNED_TASK_ID","expected_version":1,"changes":{"current_stage":"inspect","interpretation":"Concise visual hypothesis, source/frame citations and uncertainty","questions":[{"id":"relation","question":"Which relation does the end frame support?","affects_action":true,"source":"demo"}]}}

questions, current_stage and interpretation belong INSIDE changes, not at top level. Omit query entirely for visual questions. Numeric result/state/evidence fields are backend-owned. Update interpretation when new evidence resolves an alternative. Re-read after version conflicts; never overwrite a newer plan. Read back the final task before reporting completion or submitting an offline program.

## Live metric evidence

A demonstration explains procedure; its pixels do not locate today's objects. Read demo_reference(overview=true), then camera_observe and robot_status for current evidence before choosing physical actions. Use original live images for semantics and calibrated geometry for coordinates. Supported scene_query operations are surface_point, accumulate_region, fit_plane, point_distance and point_plane_distance; there is no reachability or collision-proof query.

For a numeric question, query includes operation, e.g. {"operation":"surface_point","pixel":[320,240]}. Capture with camera_observe(question_id=...,serial=...), or use scene_query on an explicit observation_path. A question-linked direct query also needs the current expected_version. Select the camera that supplied the pixels. Check frame, metres, calibration identity, time, support and residual; optical-frame coordinates are not robot targets, residual and voxel size are not total accuracy.

If a view is insufficient, propose a useful new view through existing approved movement tools, then observe again. Observe and scene_query never move the robot. For stationary objects, accumulate calibrated stationary-view ROIs; use a new batch_id after changing the regions or scene. Invisible space remains unknown. Neither a plane nor a surface point proves a clear handle opening, completed insertion or contact.

On resume, refresh live/geometry evidence that became stale after movement, object changes or calibration changes. Historical measurements retain their historical scope. Success requires current outcome evidence. Existing approval, limits and request_id rules still apply.

During online execution, a plan is not a completed action. Invoke the supplied robot primitives with measured parameters, inspect their actual result, then inspect a new frame before deciding the next motion. Do not put several motion commands into one model response when their parameters depend on earlier results. A rejected target can be revised using new evidence; an unknown execution outcome must not be retried or followed by another motion. Report the observed outcome and remaining uncertainty rather than submitting a symbolic program in place of execution.

When harness_review is supplied, treat it as an advisory about public tool behavior, not a success verdict or motion permission. For repeated queries or rejected targets, cite the existing result and explain what changed before repeating it; revise the consequential interpretation rather than only renaming the stage. A completion reference must include a successful execution receipt and current outcome evidence; a blocked/rejected/unknown request alone cannot establish completion. Plans remain revisable hypotheses. A request for independent review is a suggestion, not evidence that another model has been called.

动作后的进度应在收到实际结果后，使用最新 task_id/version 单独写回；不要在尚未收到动作结果的同一批次预写完成。版本冲突只重新核对和写 plan，不重放动作。步骤 ID 保持稳定，措辞可以随证据修正。
