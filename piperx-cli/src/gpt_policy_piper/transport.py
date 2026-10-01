"""Loopback simulation connection; never reconnect or replay unknown writes."""
from __future__ import annotations
import threading
import time
from pathlib import Path
from piperx_middleware.client import RobotClient


class BridgeError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(str(result.get('error', result)))


def checked(result):
    if 'error' in result:
        raise BridgeError(result)
    return result


class SimulationTransport:
    def __init__(self, settings, *, control=False, client_factory=RobotClient):
        config = settings['piper_simulation']
        self.client = client_factory(config['url'], Path(config['model_token_file']))
        self.operator = None
        self.session_id = None
        self._stop = threading.Event()
        self._heartbeat_error = None
        self._thread = None
        try:
            cap = checked(self.client.call('GET', '/v1/capabilities'))
            if cap.get('backend') != 'mujoco' or not cap.get('simulation'):
                raise ValueError('GPT-Policy bridge accepts only an existing MuJoCo simulator')
            self.instance_id = cap['instance_id']
            if control:
                self.operator = client_factory(config['url'], Path(config['operator_token_file']))
                session = checked(self.operator.call('POST', '/operator/session', {'owner': 'gpt-policy-piper', 'shutdown_on_loss': False}))
                self.session_id = session['session_id']
                self.client.http.headers['X-Piper-Control-Session'] = self.session_id
                self._thread = threading.Thread(target=self._heartbeat, name='piper-gpt-session', daemon=True)
                self._thread.start()
        except Exception:
            self.close()
            raise

    def _heartbeat(self):
        while not self._stop.wait(1.):
            try:
                checked(self.operator.call('POST', '/operator/session/heartbeat', {'session_id': self.session_id}))
            except Exception as exc:
                self._heartbeat_error = exc
                return

    def call(self, method, path, body=None):
        if method != 'GET' and path != '/v1/stop' and self._heartbeat_error is not None:
            raise RuntimeError('Control-session heartbeat failed; no new actions admitted') from self._heartbeat_error
        result = checked(self.client.call(method, path, body))
        if result.get('instance_id', self.instance_id) != self.instance_id:
            raise RuntimeError('Simulation executor restarted; reopen the bridge deliberately')
        return result

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12)
        if self.operator is not None:
            if self.session_id:
                self.operator.call('POST', '/operator/session/release', {'session_id': self.session_id})
            self.operator.close()
            self.operator = None
        if getattr(self, 'client', None) is not None:
            self.client.close()
