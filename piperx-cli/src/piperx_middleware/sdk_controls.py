"""Pinned Piper v189 parameter transactions, with exact frame admission.

No arbitrary CAN, enable, reset, zeroing or force-mode escape hatch. SDK ACK is
not physical verification. Partial configuration is explicitly reported.
"""
import math
import struct
import time

from .models import DomainError


class SDKControls:
    def query_limits(self):
        records = []
        for i in range(1, 7):
            # Clear the SDK cache BEFORE querying; stale cached values are not readback.
            cached = getattr(self.arm._parser, "motor_angle_limit_max_spd", None)
            if cached is not None:
                cached.msg.joints[i-1].clear()
            began = time.monotonic()
            with self._transaction([(0x472, bytes([i, 1, 0, 0, 0, 0, 0, 0]))]):
                value = self.arm.get_joint_angle_vel_limits(i, timeout=.2, min_interval=0.)
            receipt = self.last_seen.get(0x473)
            if value is None or receipt is None or receipt < began:
                records.append({"joint": i, "available": False, "reason": "query_timeout"})
                continue
            m = value.msg
            values = [m.min_angle_limit, m.max_angle_limit, m.max_joint_spd]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                records.append({"joint": i, "available": False, "reason": "invalid_feedback"})
                continue
            records.append({"joint": i, "available": True, "min_deg": math.degrees(values[0]),
                "max_deg": math.degrees(values[1]), "max_velocity_deg_s": math.degrees(values[2]),
                "received_at_monotonic_s": receipt, "source": "sdk_query_response"})
        return {"available": all(r["available"] for r in records), "joints": records,
                "clock_domain": "executor_host_monotonic", "physical_validation": False}

    def apply_parameters(self, values):
        results = []
        for name, value in values.items():
            try:
                if name == "payload":
                    code = {"empty": 1, "half": 2, "full": 3}[value]
                    expected = (0x477, bytes([0, 0, 0, 0xAE, code, 0, 0, 0]))
                    with self._transaction([expected]):
                        ok = self.arm.set_payload(value, timeout=.2)
                    results.append({"parameter": name, "value": value, "acknowledged": bool(ok),
                                    "readback_verified": False, "status": "acknowledged" if ok else "outcome_unknown"})
                elif name == "collision_rating":
                    # All joints at once; no dependence on potentially stale current ratings.
                    with self._transaction([(0x47A, bytes([value]*6+[0, 0]))]):
                        self.arm._send_msg(self.arm._MSG_CrashProtectionRatingConfig(*([value]*6)))
                    cached = getattr(self.arm._parser, "crash_protection_rating", None)
                    if cached is not None: cached.msg.clear()
                    began = time.monotonic()
                    with self._transaction([(0x477, bytes([2, 0, 0, 0, 3, 0, 0, 0]))]):
                        readback = self.arm.get_crash_protection_rating(timeout=.2, min_interval=0.)
                    ok = (readback is not None and readback.msg == [value]*6
                          and self.last_seen.get(0x47B, -1) >= began)
                    results.append({"parameter": name, "value": value, "readback_verified": ok,
                                    "status": "verified_readback" if ok else "outcome_unknown"})
                elif name == "joint_acc_rad_s2":
                    ticks = round(value*100)
                    for i in range(1, 7):
                        expected = (0x475, bytes([i, 0, 0xAE])+struct.pack(">H", ticks)+bytes(3))
                        with self._transaction([expected]):
                            self.arm._send_msg(self.arm._MSG_JointConfig(joint_index=i,
                                acc_param_config_is_effective_or_not=0xAE, max_joint_acc=ticks))
                        cache = getattr(self.arm._parser, "motor_max_acc_limit", None)
                        if cache is not None: cache.msg.joints[i-1].clear()
                        began = time.monotonic()
                        with self._transaction([(0x472, bytes([i, 2, 0, 0, 0, 0, 0, 0]))]):
                            rb = self.arm.get_joint_acc_limits(i, timeout=.2, min_interval=0.)
                        ok = rb is not None and round(rb.msg.max_joint_acc*100) == ticks and self.last_seen.get(0x47C, -1) >= began
                        results.append({"parameter": name, "joint": i, "value": ticks/100,
                                        "readback_verified": ok, "status": "verified_readback" if ok else "outcome_unknown"})
                        if not ok: break
                else:
                    raise DomainError("unsupported_operation", "Unrecognized firmware parameter.", 422)
            except Exception as exc:
                results.append({"parameter": name, "status": "outcome_unknown", "error": str(exc)})
            if any(r["status"] == "outcome_unknown" for r in results): break
        return {"status": "outcome_unknown" if any(r["status"] == "outcome_unknown" for r in results) else "applied",
                "results": results, "physical_validation": False,
                "message": "Controller ACK/readback only. A partial or unknown result must be inspected, never automatically replayed."}
