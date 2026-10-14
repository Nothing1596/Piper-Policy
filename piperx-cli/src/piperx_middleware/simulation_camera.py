"""Current simulation evidence shared by the independent CLI and MCP."""
import base64
import json
from pathlib import Path
from mcp.server.fastmcp.exceptions import ToolError


def capture(client, workspace, output):
    root = Path(workspace).resolve(strict=True)
    destination = (root / output).resolve()
    if not root.is_dir() or destination == root or not destination.is_relative_to(root):
        raise ToolError('Path must be inside the configured workspace')
    if destination.exists():
        raise ToolError('Output already exists; choose a new path')
    capabilities = client.call('GET', '/v1/capabilities')
    if capabilities.get('backend') != 'mujoco':
        raise ToolError('This camera tool requires the MuJoCo simulator')
    observation = dict(client.call('GET', '/v1/simulation/observation'))
    if 'error' in observation:
        raise ToolError('Simulation observation unavailable')
    rgb = base64.b64decode(observation.pop('rgb_jpeg_b64'), validate=True)
    depth = base64.b64decode(observation.pop('depth_npy_b64'), validate=True)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'rgb.jpg').write_bytes(rgb)
    (destination / 'depth.npy').write_bytes(depth)
    (destination / 'observation.json').write_text(json.dumps(observation, indent=2), encoding='utf-8')
    return {'evidence': str(destination), **observation}
