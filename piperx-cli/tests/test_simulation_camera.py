import asyncio
import pytest
from piperx_middleware.mcp_server import create_mcp
from piperx_middleware.simulation_camera import capture


class NoIO:
    def call(self, *args):
        raise AssertionError('Invalid destinations must not contact executor')


@pytest.mark.parametrize('output', ['.', '../escape', 'existing'])
def test_capture_rejects_unsafe_or_existing_destination(tmp_path, output):
    (tmp_path/'existing').mkdir()
    with pytest.raises(Exception, match='workspace|already exists'):
        capture(NoIO(), tmp_path, output)


def test_robot_camera_opt_in(tmp_path):
    async def run():
        plain = {t.name for t in await create_mcp(NoIO()).list_tools()}
        camera = {t.name for t in await create_mcp(NoIO(), workspace=tmp_path).list_tools()}
        assert 'simulation_observe' not in plain
        assert camera - plain == {'simulation_observe'}
        assert 'video_compile' not in camera
    asyncio.run(run())
