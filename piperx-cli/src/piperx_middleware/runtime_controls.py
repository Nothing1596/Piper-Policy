"""Operator controls and supervision; model tools cannot clear latches or configure safety."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .models import DomainError, JOINT_LIMITS_DEG


class RuntimeControls:
    def _init_controls(self):
        self._estop_file = Path(self.settings.data_dir) / "estop.latched"
        self.estop_latched = self._estop_file.exists()
        self.backend.estopped = self.estop_latched
        self.parameter_version = 0
        self.measured_limits = None
        self._supervisor_exit = threading.Event()
        self._supervisor = threading.Thread(target=self._supervise, name="piperx-safety", daemon=True)
        self._supervisor.start()

    def _supervise(self):
        # Does not acquire service.lock: a model request or IK cannot stall supervision.
        while not self._supervisor_exit.wait(0.05):
            if self.estop_latched or self.closed:
                continue
            try:
                sample = self.backend.snapshot()
                if sample.connected and sample.collision_status and any(sample.collision_status):
                    self.emergency_stop("collision_feedback")
            except Exception:
                if self.active:
                    self.stop()  # Worker rejects unknown/stale feedback; no invented measurements.

    def emergency_stop(self, reason="operator"):
        # Latch/cancel precedes journal or transport I/O. Never waits for the motion lock.
        self.estop_latched = True
        self.backend.estopped = True
        self.stop()
        persistence_error = None
        try:
            with self._estop_file.open("w", encoding="utf-8") as f:
                f.write(reason)
                f.flush()
                import os
                os.fsync(f.fileno())
        except OSError as exc:
            persistence_error = type(exc).__name__
        try:
            self.backend.emergency_stop()
            result = {"status": "software_estop_sent", "latched": True,
                      "confirmed_stopped": False, "simulation": self.backend.name in ("sim", "mujoco")}
        except Exception as exc:
            result = {"status": "outcome_unknown", "latched": True, "confirmed_stopped": False,
                      "error_code": getattr(exc, "code", "stop_transport_error"), "error": str(exc)}
        result["persistence_error"] = persistence_error
        try:
            self.store.event("software_estop", reason=reason, **result)
        except Exception:
            result["audit_error"] = "unavailable"
        return result

    def clear_estop(self):
        with self.lock:
            self._require_open()
            if self.active:
                raise DomainError("busy", "Wait for the interrupted job to finish.")
            sample = self.backend.snapshot()
            # Clear ONLY our latch. Never send firmware reset/enable/homing.
            sample.diagnostics.pop("software_estop_latched", None)
            from .backends import check_state
            check_state(sample, self.settings.feedback_timeout_s)
            self._estop_file.unlink(missing_ok=True)
            self.estop_latched = self.backend.estopped = False
            self.plans.clear()
            self.epoch += 1
            self.lease = None
            self.store.event("software_estop_cleared", connection_epoch=self.epoch)
            return {"status": "latch_cleared", "firmware_reset_sent": False, "motion_sent": False}

    def diagnostics(self):
        state = self.state()
        return {"backend": self.backend.name, "simulation": self.backend.name in ("sim", "mujoco"),
                "estop_latched": self.estop_latched, "parameter_version": self.parameter_version,
                "supervisor_alive": self._supervisor.is_alive(), "state": state,
                "joint_limits": {"configured_deg": JOINT_LIMITS_DEG,
                    "measured": self.measured_limits, "available": bool(self.measured_limits and self.measured_limits.get("available")),
                    "reason": None if self.measured_limits and self.measured_limits.get("available") else
                              "explicit_query_required" if self.backend.name == "agx" else "no_physical_hardware"},
                "physical_validation": False}

    def limits(self, refresh=False):
        with self.lock:
            self._require_open()
            if refresh:
                if self.active:
                    raise DomainError("busy", "Limit queries require an idle executor.")
                if self.backend.name in ("sim", "mujoco"):
                    self.measured_limits = {"available": False, "reason": "no_physical_hardware", "simulation": True}
                else:
                    if not self.backend.snapshot().connected:
                        raise DomainError("not_connected", "Connect before querying hardware limits.")
                    self.measured_limits = self.backend.query_limits()
                    self.measured_limits["connection_epoch"] = self.epoch
                self.store.event("limits_query", result=self.measured_limits)
            return {"configured_deg": JOINT_LIMITS_DEG, "measured": self.measured_limits,
                    "connection_epoch": self.epoch, "queried": refresh}

    def configure_runtime(self, request):
        values = request.model_dump(exclude_none=True)
        if not values:
            raise DomainError("invalid_request", "No runtime parameter specified.", 422)
        with self.lock:
            self._require_open()
            if self.active:
                raise DomainError("busy", "Runtime parameters require an idle executor.")
            if not self.settings.allow_motion:
                raise DomainError("read_only", "Runtime changes are disabled.", 403)
            self._check(self.backend.snapshot())
            hardware = {k: v for k, v in values.items() if not k.startswith("tcp_offset")}
            if hardware and self.backend.name == "mujoco":
                raise DomainError("unsupported_sim_parameter", "Firmware parameter emulation is not provided by MuJoCo", 422)
            firmware_result = None
            if hardware and self.backend.name != "sim":
                self.plans.clear()
                self.lease = None
                firmware_result = self.backend.apply_parameters(hardware)
                self.store.event("firmware_parameters", result=firmware_result)
                if firmware_result["status"] != "applied":
                    return firmware_result
            for name in ("tcp_offset_m", "tcp_offset_rpy_deg"):
                if name in values:
                    setattr(self.settings, name, values[name])
            if hardware and self.backend.name == "sim":
                self.backend.runtime_parameters.update(hardware)
            self.parameter_version += 1
            self.measured_limits = None
            self.plans.clear()
            self.lease = None
            self.epoch += 1
            self.store.event("runtime_parameters", values=values, parameter_version=self.parameter_version)
            return {"status": "applied", "configured": values, "parameter_version": self.parameter_version,
                    "source": "synthetic" if self.backend.name == "sim" else "host_runtime",
                    "hardware_verified": False, "persistent": False, "firmware_result": firmware_result}

    def inject_fault(self, request):
        with self.lock:
            self._require_open()
            if self.backend.name != "sim":
                raise DomainError("simulation_only", "Fault injection is only available in simulation.", 403)
            with self.backend.lock:
                self.backend.stale = request.fault == "stale"
                self.backend.collision = request.fault == "collision"
                self.backend.follow = request.fault != "tracking"
                self.backend.fault = request.fault == "driver"
                self.backend.teach = request.fault == "teaching"
            self.store.event("simulation_fault", fault=request.fault)
            return {"simulation": True, "fault": request.fault}
