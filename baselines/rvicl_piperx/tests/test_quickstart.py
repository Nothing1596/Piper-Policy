"""The release launcher must preserve downloads and reject corrupt evidence."""
import hashlib
from pathlib import Path

import pytest

import quickstart


def test_verified_cache_needs_no_network(tmp_path, monkeypatch):
    target = tmp_path / 'resource'
    target.write_bytes(b'original')
    monkeypatch.setattr(quickstart.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('cache must not download'))
    quickstart.fetch(target, {'url': 'https://example.invalid', 'sha256': quickstart.digest(target)})


def test_changed_cache_is_preserved(tmp_path):
    target = tmp_path / 'resource'
    target.write_bytes(b'user-changed')
    with pytest.raises(ValueError, match='retained for inspection'):
        quickstart.fetch(target, {'url': 'https://example.invalid', 'sha256': '0' * 64})
    assert target.read_bytes() == b'user-changed'


def test_invalid_download_is_not_installed(tmp_path):
    source = tmp_path / 'upstream'
    source.write_bytes(b'wrong upstream bytes')
    target = tmp_path / 'cache/resource'
    with pytest.raises(ValueError, match='Downloaded file hash mismatch'):
        quickstart.fetch(target, {'url': source.as_uri(), 'sha256': '0' * 64})
    assert not target.exists()
    assert not list(target.parent.glob('*.partial'))


def test_valid_download_can_be_reused(tmp_path):
    source = tmp_path / 'upstream'
    source.write_bytes(b'fixed upstream bytes')
    target = tmp_path / 'cache/resource'
    spec = {'url': source.as_uri(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    quickstart.fetch(target, spec)
    source.unlink()
    quickstart.fetch(target, spec)
    assert target.read_bytes() == b'fixed upstream bytes'


def test_native_windows_rejected_before_startup(monkeypatch):
    monkeypatch.setattr(quickstart.sys, 'platform', 'win32')
    with pytest.raises(SystemExit) as error:
        quickstart.main(['run'])
    assert error.value.code == 2


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', quickstart.os.environ.get('PATH', ''))
    monkeypatch.setattr(quickstart, 'ROOT', tmp_path)
    monkeypatch.setattr(quickstart.sys, 'prefix', str(tmp_path / '.venv'))
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    import rvicl_ig.piper_cli
    monkeypatch.setattr(rvicl_ig.piper_cli, 'verify_sources', lambda *a: None)
    monkeypatch.setattr(quickstart, 'prepare', lambda *a: pytest.fail('custom tasks must not prepare bundled videos'))
    calls = []
    monkeypatch.setattr(quickstart.subprocess, 'call', lambda args, **kw: calls.append((args, kw)) or 0)
    return tmp_path, calls


def test_custom_manifest_and_overrides_reach_existing_batch(launcher):
    root, calls = launcher
    manifest = root / 'my-tasks.json'
    manifest.write_text('[]')
    assert quickstart.main(['run', '--manifest', str(manifest), '--dry-run', '--model', 'my-model',
        '--reasoning-effort', 'high', '--sam3-endpoint', 'http://127.0.0.1:9999',
        '--max-turns', '37', '--max-episode-steps', '600', '--primitive-image-history', 'budget']) == 0
    args, kwargs = calls[0]
    for flag, value in [('--manifest', str(manifest)), ('--model', 'my-model'),
            ('--reasoning-effort', 'high'), ('--max-turns', '37'), ('--max-episode-steps', '600'),
            ('--sam3-endpoint', 'http://127.0.0.1:9999'), ('--primitive-image-history', 'budget')]:
        assert args[args.index(flag) + 1] == value
    assert '--dry-run' in args and '--env-file' not in args
    assert kwargs['env']['RPENT_ROOT'] == str(root / 'deps/RPent')


def test_custom_config_env_and_output_are_not_replaced(launcher):
    root, calls = launcher
    for name in ['manifest.json', 'custom.json', 'private.env']:
        (root / name).write_text('{}')
    assert quickstart.main(['run', '--manifest', str(root / 'manifest.json'),
        '--config', str(root / 'custom.json'), '--env-file', str(root / 'private.env'),
        '--output', str(root / 'independent-output'), '--ffmpeg-dir', str(root / 'ffmpeg')]) == 0
    args, kwargs = calls[0]
    for flag, name in [('--config', 'custom.json'), ('--env-file', 'private.env'),
            ('--output-root', 'independent-output'), ('--ffmpeg-dir', 'ffmpeg')]:
        assert args[args.index(flag) + 1] == str(root / name)
    assert str(root / 'ffmpeg') in kwargs['env']['PATH'].split(quickstart.os.pathsep)


def test_missing_custom_manifest_stops_without_fallback(launcher):
    root, calls = launcher
    with pytest.raises(SystemExit) as error:
        quickstart.main(['run', '--manifest', str(root / 'missing.json'), '--dry-run'])
    assert error.value.code == 2 and calls == []


def test_model_run_requires_private_env_before_starting(launcher):
    root, calls = launcher
    manifest = root / 'tasks.json'
    manifest.write_text('[]')
    with pytest.raises(SystemExit) as error:
        quickstart.main(['run', '--manifest', str(manifest)])
    assert error.value.code == 2 and calls == []


def test_default_examples_still_use_prepared_manifest(launcher, monkeypatch):
    root, calls = launcher
    directory = root / 'var/watchact-v5'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text('[]')
    verified = []
    monkeypatch.setattr(quickstart, 'prepare', lambda path: verified.append(path))
    assert quickstart.main(['run', '--dry-run']) == 0
    args = calls[0][0]
    assert args[args.index('--manifest') + 1] == str(directory / 'manifest.json')
    assert verified == [directory]


def test_custom_manifest_not_silently_ignored_by_prepare(launcher):
    root, calls = launcher
    with pytest.raises(SystemExit) as error:
        quickstart.main(['prepare', '--manifest', str(root / 'tasks.json')])
    assert error.value.code == 2 and calls == []
