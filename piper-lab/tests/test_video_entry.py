import pytest
from piperlab.video_entry import main


@pytest.mark.parametrize('args', [['sim'], ['policy'], ['mcp', '--workspace', '.', '--robot-root', 'sim']])
def test_video_entry_cannot_enable_robot_tools(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
