import time

import pytest
from fastapi.testclient import TestClient

from piperx_middleware.backends import SimBackend
from piperx_middleware.http_api import create_app
from piperx_middleware.models import (DomainError, ExecuteRequest, JointMove, MoveLinear,
                                    RuntimeParameters, Settings, SimFault)
from piperx_middleware.service import RobotService
from piperx_middleware.kinematics import PiperKinematics


@pytest.fixture
def service(tmp_path):
    s = RobotService(SimBackend(), Settings(data_dir=tmp_path))
    s.connect()
    yield s
    s.close()


def wait(s, job):
    until = time.monotonic() + 10
    while job["status"] in {"accepted", "running"}:
        assert time.monotonic() < until
        time.sleep(.01)
        job = s.get_job(job["job_id"])
    return job


def test_completion_and_idempotency(service):
    c = JointMove(joints_deg=[1., 2., -3., 0., 0., 0.])
    j = wait(service, service.move(c, "test-motion-0001"))
    assert j["status"] == "succeeded"
    assert j["after"]["q_deg"] == c.joints_deg
    assert j["simulation"] and not j["physical_validation"]
    count = len(service.backend.commands)
    assert service.move(c, "test-motion-0001")["job_id"] == j["job_id"]
    assert len(service.backend.commands) == count
    with pytest.raises(DomainError, match="another command"):
        service.move(JointMove(joints_deg=[0.]*6), "test-motion-0001")


def test_restart_does_not_replay(tmp_path):
    settings = Settings(data_dir=tmp_path)
    s = RobotService(SimBackend(), settings)
    s.connect()
    cmd = JointMove(joints_deg=[1., 0., 0., 0., 0., 0.])
    job = wait(s, s.move(cmd, "restart-motion-1"))
    s.close()
    s = RobotService(SimBackend(), settings)
    try:
        assert s.move(cmd, "restart-motion-1")["job_id"] == job["job_id"]
        assert not s.backend.commands
    finally:
        s.close()


@pytest.mark.parametrize("fault,code", [("stale", "stale_feedback"), ("driver", "robot_not_ready"), ("teaching", "control_mode")])
def test_admission_fault(service, fault, code):
    service.inject_fault(SimFault(fault=fault))
    with pytest.raises(DomainError) as exc:
        service.move(JointMove(joints_deg=[0.]*6), "fault-motion-001")
    assert exc.value.code == code
    assert not service.backend.commands


def test_timeout_not_success(service):
    service.backend.follow = False
    j = wait(service, service.move(JointMove(joints_deg=[2., 0., 0., 0., 0., 0.], timeout_s=.15), "tracking-motion-1"))
    assert j["status"] == "outcome_unknown"
    assert j["error_code"] == "execution_timeout"


def test_stop_invalidates_plan(service):
    p = service.preview(JointMove(joints_deg=[0.]*6))
    service.stop()
    with pytest.raises(DomainError) as exc:
        service.execute(ExecuteRequest(plan_id=p["plan_id"], request_id="old-plan-test-1"))
    assert exc.value.code == "cancelled"


def test_estop_persists_and_operator_clear(tmp_path):
    settings = Settings(data_dir=tmp_path)
    s = RobotService(SimBackend(), settings)
    s.connect()
    assert s.emergency_stop()["latched"]
    s.close()
    s = RobotService(SimBackend(), settings)
    try:
        s.connect()
        assert not s.state()["ready"]
        with pytest.raises(DomainError) as exc:
            s.move(JointMove(joints_deg=[0.]*6), "stop-motion-001")
        assert exc.value.code == "estop_latched"
        assert not s.clear_estop()["motion_sent"]
        assert s.state()["ready"]
    finally:
        s.close()


def test_collision_supervisor_independent(service):
    # Holding the service lock must not delay the independent collision latch.
    with service.lock:
        service.backend.collision = True
        until = time.monotonic()+1
        while not service.estop_latched and time.monotonic() < until:
            time.sleep(.01)
        assert service.estop_latched
    with pytest.raises(DomainError): service.clear_estop()
    service.inject_fault(SimFault(fault="none"))
    service.clear_estop()


def test_tcp_configuration_invalidates_plan(service):
    p = service.preview(JointMove(joints_deg=[0.]*6))
    service.configure_runtime(RuntimeParameters(tcp_offset_m=[0., 0., .02]))
    with pytest.raises(DomainError) as exc:
        service.execute(ExecuteRequest(plan_id=p["plan_id"], request_id="old-tcp-plan-1"))
    assert exc.value.code == "expired_plan"


def test_linear_path_sampled_and_executed(service):
    service.backend.q = [0., 45., -60., 0., 20., 0.]
    kin = PiperKinematics()
    start = kin.pose(service.backend.q)
    target = list(start["xyz_m"])
    target[2] += .003
    cmd = MoveLinear(xyz_m=target, step_m=.001)
    plan = service.preview_primitive(cmd)
    points = plan["resolution"]["waypoints_deg"]
    for i, q in enumerate(points, 1):
        pose = kin.pose(q)
        expected = [(b-a)*i/len(points)+a for a,b in zip(start["xyz_m"], target)]
        assert max(abs(a-b) for a,b in zip(pose["xyz_m"], expected)) < 1e-5
    job = wait(service, service.execute(ExecuteRequest(plan_id=plan["plan_id"], request_id="line-test-001")))
    assert job["status"] == "succeeded"
    assert len([c for c in service.backend.commands if c[0] == "joint"]) >= len(points)


def test_linear_unreachable_no_write(service):
    with pytest.raises(DomainError):
        service.primitive(MoveLinear(xyz_m=[10., 10., 10.]), "unreachable-001")
    assert not service.backend.commands


def test_role_boundary_and_schemas(service):
    model, operator = "m"*40, "o"*40
    with TestClient(create_app(service, model, operator)) as c:
        mh, oh = {"Authorization": "Bearer "+model}, {"Authorization": "Bearer "+operator}
        assert c.post("/operator/estop", headers=mh).status_code == 401
        assert c.patch("/operator/parameters", headers=mh, json={"payload": "full"}).status_code == 401
        assert c.get("/v1/diagnostics", headers=mh).status_code == 200
        assert c.post("/v1/move", headers=mh, json={"command": {"kind": "joint", "joints_deg": [999.]*6}, "request_id": "bounds-test-1"}).status_code == 422
        assert c.post("/operator/estop", headers=oh).json()["latched"]
        assert c.post("/operator/clear-estop", headers=oh).status_code == 200


def test_connect_error_structured(tmp_path):
    class MissingCAN(SimBackend):
        def connect(self): raise RuntimeError("no adapter")
    s = RobotService(MissingCAN(), Settings(data_dir=tmp_path))
    try:
        with pytest.raises(DomainError) as exc: s.connect()
        assert exc.value.code == "can_unavailable" and exc.value.status == 503
    finally: s.close()


def test_native_service_verifies_tcp_and_requires_explicit_return_to_joint_mode(tmp_path):
    # Offline behavioural adapter, not physical hardware evidence.
    from piperx_middleware.models import ControlMode
    class NativeFixture(SimBackend):
        name = "agx"
        def linear_target(self, current, pose, speed, checkpoint):
            checkpoint()
            self.motion_mode = 2
            self.q = PiperKinematics().solve(pose["xyz_m"], pose["rpy_deg"], current)["joints_deg"]
            self.commands.append(("native_linear", pose))
    b = NativeFixture()
    b.q = [0.,45.,-60.,0.,20.,0.]
    s = RobotService(b, Settings(backend="agx", data_dir=tmp_path))
    try:
        s.connect()
        xyz = PiperKinematics().pose(b.q)["xyz_m"]
        xyz[2] += .003
        j = wait(s, s.primitive(MoveLinear(xyz_m=xyz, native_controller=True), "native-fixture-1"))
        assert j["status"] == "succeeded", j
        assert j["resolution"]["path_semantics"] == "controller_move_l"
        assert s.state()["not_ready_reason"]["code"] == "control_mode"
        assert wait(s, s.move(ControlMode(), "return-mode-001"))["status"] == "succeeded"
    finally: s.close()


def test_readonly_rejects_commands(service):
    service.settings.allow_motion = False
    with pytest.raises(DomainError) as exc:
        service.move(JointMove(joints_deg=[0.]*6), "readonly-test-1")
    assert exc.value.code == "read_only"
    assert not service.backend.commands


def test_freshness_requires_source_stamp(service):
    from piperx_middleware.backends import check_state
    state = service.backend.snapshot()
    state.feedback_stamp_s = None
    with pytest.raises(DomainError): check_state(state, .15)
    state = service.backend.snapshot()
    state.feedback_age_s = -.1
    with pytest.raises(DomainError): check_state(state, .15)


def test_queried_limits_narrow_configured_bounds(service):
    service.measured_limits = {"available": True, "connection_epoch": service.epoch,
                              "joints": [{"min_deg": -1., "max_deg": 1.}]*6}
    with pytest.raises(DomainError) as exc:
        service.move(JointMove(joints_deg=[2.,0.,0.,0.,0.,0.]), "measured-test-1")
    assert exc.value.code == "measured_joint_limits"
