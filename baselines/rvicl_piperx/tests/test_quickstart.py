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
