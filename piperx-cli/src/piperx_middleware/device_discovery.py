from __future__ import annotations

import ctypes
from pathlib import Path
import sys
import tempfile
from typing import Any

from .models import DomainError, Settings
from .process_lock import ProcessLock

# Test injection helpers
_cando_dll_override: Any | None = None
_sys_class_net_override: Path | None = None


def set_cando_dll(dll: Any | None) -> None:
    """Inject a fake DLL or mock for testing CANDO enumeration."""
    global _cando_dll_override
    _cando_dll_override = dll


def set_sys_class_net_path(path: Path | None) -> None:
    """Inject an alternative /sys/class/net directory for Linux SocketCAN mock testing."""
    global _sys_class_net_override
    _sys_class_net_override = path


def get_cando_dll(settings: Settings | None = None) -> Any:
    """Load CANDO DLL on demand without loading on import or on mac client."""
    global _cando_dll_override
    if _cando_dll_override is not None:
        return _cando_dll_override

    if sys.platform != "win32":
        raise DomainError(
            "platform_unsupported",
            f"CANDO DLL is only available on Windows (current platform: {sys.platform})",
        )

    if settings and getattr(settings, "cando_source", None):
        source = str(settings.cando_source)
        if source not in sys.path:
            sys.path.insert(0, source)

    try:
        from agx_cando.dll import load_cando_dll

        return load_cando_dll()
    except Exception as exc:
        raise DomainError("dll_load_failed", f"Failed to load CANDO DLL: {exc}") from exc


def _bind_scan_functions(dll: Any) -> None:
    """Bind ONLY list_* functions. Never bind or call open/describe."""
    _BOOL = ctypes.c_bool
    _U8 = ctypes.c_uint8
    _VOID_P = ctypes.c_void_p

    bindings = [
        ("cando_list_malloc", [ctypes.POINTER(_VOID_P)], _BOOL),
        ("cando_list_free", [_VOID_P], _BOOL),
        ("cando_list_scan", [_VOID_P], _BOOL),
        ("cando_list_num", [_VOID_P, ctypes.POINTER(_U8)], _BOOL),
    ]
    for name, argtypes, restype in bindings:
        fn = getattr(dll, name, None)
        if fn is not None and hasattr(fn, "argtypes"):
            try:
                fn.argtypes = argtypes
                fn.restype = restype
            except (AttributeError, TypeError):
                pass


def _is_backend_connected(service: Any) -> bool:
    """Check if the service backend is already connected without sending frames or opening devices."""
    if hasattr(service, "is_connected"):
        val = service.is_connected
        return val() if callable(val) else bool(val)

    backend = getattr(service, "backend", None)
    if backend is not None:
        if hasattr(backend, "is_connected"):
            val = backend.is_connected
            return val() if callable(val) else bool(val)
        if hasattr(backend, "arm") and backend.arm is not None:
            is_c = getattr(backend.arm, "is_connected", None)
            if callable(is_c):
                return bool(is_c())
        if hasattr(backend, "snapshot") and callable(backend.snapshot):
            try:
                return bool(getattr(backend.snapshot(), "connected", False))
            except Exception:
                pass
    return False


def _scan_socketcan(sys_class_net_dir: Path | None = None) -> list[dict[str, Any]]:
    """Enumerate Linux /sys/class/net type 280 (ARPHRD_CAN) interfaces with no link setup."""
    net_dir = sys_class_net_dir or _sys_class_net_override or Path("/sys/class/net")
    devices: list[dict[str, Any]] = []

    if not net_dir.exists() or not net_dir.is_dir():
        return devices

    try:
        entries = sorted(net_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return devices

    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            type_file = entry / "type"
            if not type_file.exists():
                continue
            if type_file.read_text().strip() == "280":
                name = entry.name
                devices.append({
                    "id": f"socketcan:{name}",
                    "label": f"SocketCAN {name}",
                    "interface": "socketcan",
                    "channel": name,
                    "device_index": None,
                    "connected": False,
                })
        except OSError:
            continue

    return devices


class _ScanResources:
    def __init__(self, lock):
        self.lock = lock
        self.dll = None
        self.handle = ctypes.c_void_p()

    def close(self):
        if self.handle:
            try:
                ok = self.dll.cando_list_free(self.handle)
            except Exception as exc:
                raise DomainError("cleanup_incomplete", "CANDO discovery list cleanup failed; resources retained.") from exc
            if not ok:
                raise DomainError("cleanup_incomplete", "CANDO discovery list cleanup failed; resources retained.")
            self.handle = ctypes.c_void_p()
        self.lock.close()


def cleanup_discovery(service):
    """Retry retained discovery-list cleanup before another scan/open or shutdown."""
    pending = getattr(service, "_discovery_scan", None)
    if pending is not None:
        pending.close()
        service._discovery_scan = None


def _scan_cando(service) -> list[dict[str, Any]]:
    cleanup_discovery(service)
    lock_path = Path(tempfile.gettempdir()) / "piperx-can-agx_cando-0.lock"
    try:
        resources = _ScanResources(ProcessLock(lock_path))
    except (RuntimeError, OSError) as exc:
        raise DomainError("hardware_busy", "Another middleware process owns CANDO adapter.") from exc
    service._discovery_scan = resources
    try:
        dll = resources.dll = get_cando_dll(service.settings)
        _bind_scan_functions(dll)
        for name, args in (
            ("cando_list_malloc", (ctypes.byref(resources.handle),)),
            ("cando_list_scan", (resources.handle,)),
        ):
            try:
                ok = getattr(dll, name)(*args)
            except Exception as exc:
                raise DomainError("scan_failed", f"{name} raised {type(exc).__name__}") from exc
            if not ok or not resources.handle:
                raise DomainError("scan_failed", f"{name} failed")
        count = ctypes.c_uint8()
        try:
            ok = dll.cando_list_num(resources.handle, ctypes.byref(count))
        except Exception as exc:
            raise DomainError("scan_failed", "cando_list_num raised") from exc
        if not ok:
            raise DomainError("scan_failed", "cando_list_num failed")
        return [{"id": f"cando:{i}", "label": f"CANDO adapter {i}",
                 "interface": "agx_cando", "channel": "0", "device_index": i,
                 "connected": False} for i in range(count.value)]
    finally:
        # A failed free keeps both handle and lock reachable for an explicit retry.
        cleanup_discovery(service)


def _set_cando_device_index(settings: Settings, value: int | None) -> None:
    settings.cando_device_index = value


def discover_devices(service: Any) -> dict[str, Any]:
    """Discover available PiperX devices under service.lock."""
    settings: Settings = service.settings
    backend_name = getattr(settings, "backend", "sim")
    if backend_name == "mujoco":
        connected = _is_backend_connected(service)
        return {"devices": [{"id": "mujoco:0", "label": "PiperX MuJoCo simulation",
                "interface": "mujoco", "channel": "0", "device_index": None, "connected": connected}],
                "backend": "mujoco", "connected_device_id": "mujoco:0" if connected else None,
                "notes": "Physical simulation; no hardware discovery."}

    # 1. Already connected: return current configured owner info, no DLL scan or CAN open
    if _is_backend_connected(service):
        if backend_name == "sim":
            conn_id = "sim:0"
            devices = [{
                "id": conn_id,
                "label": "PiperX Simulation",
                "interface": "sim",
                "channel": "0",
                "device_index": None,
                "connected": True,
            }]
        else:
            can_iface = getattr(settings, "can_interface", "agx_cando")
            if can_iface == "agx_cando":
                dev_idx = getattr(settings, "cando_device_index", None)
                idx_val = 0 if dev_idx is None else dev_idx
                conn_id = f"cando:{idx_val}"
                devices = [{
                    "id": conn_id,
                    "label": f"CANDO adapter {idx_val}",
                    "interface": "agx_cando",
                    "channel": str(getattr(settings, "can_channel", "0")),
                    "device_index": dev_idx,
                    "connected": True,
                }]
            else:
                ch = str(getattr(settings, "can_channel", "0"))
                conn_id = f"socketcan:{ch}"
                devices = [{
                    "id": conn_id,
                    "label": f"SocketCAN {ch}",
                    "interface": "socketcan",
                    "channel": ch,
                    "device_index": None,
                    "connected": True,
                }]

        return {
            "devices": devices,
            "backend": backend_name,
            "connected_device_id": conn_id,
            "notes": f"Reusing active session connected to {conn_id}.",
        }

    # 2. Disconnected discovery: Simulation backend
    if backend_name == "sim":
        devices = [{
            "id": "sim:0",
            "label": "PiperX Simulation",
            "interface": "sim",
            "channel": "0",
            "device_index": None,
            "connected": False,
        }]
        return {
            "devices": devices,
            "backend": "sim",
            "connected_device_id": None,
            "notes": "Simulation backend (sim:0) ready.",
        }

    # 3. Disconnected discovery: AGX backend
    can_iface = getattr(settings, "can_interface", "agx_cando")
    if can_iface == "socketcan":
        devices = _scan_socketcan()
        count = len(devices)
        if count == 0:
            notes = "No SocketCAN interfaces found."
        elif count == 1:
            notes = f"1 SocketCAN interface detected ({devices[0]['id']}). Ready to connect."
        else:
            notes = f"{count} SocketCAN interfaces detected. Specify device ID with /connect <device_id>."
        return {
            "devices": devices,
            "backend": "agx",
            "connected_device_id": None,
            "notes": notes,
        }

    devices = _scan_cando(service)
    count = len(devices)
    if count == 0:
        notes = "No CANDO adapters detected."
    elif count == 1:
        notes = "1 CANDO adapter detected (cando:0). Ephemeral index; no serial identity guarantees."
    else:
        notes = (
            f"{count} CANDO adapters detected. Discovery indices are ephemeral; "
            "no serial identity guarantees. Specify device ID with /connect <device_id>."
        )

    return {
        "devices": devices,
        "backend": "agx",
        "connected_device_id": None,
        "notes": notes,
    }


def select_device(service: Any, device_id: str) -> dict[str, Any]:
    """Validate and select a device without silent fallback."""
    # Refuses switch while connected unless caller already closed explicitly
    discovery = discover_devices(service)
    if _is_backend_connected(service):
        if device_id == discovery["connected_device_id"]:
            return {"device_id": device_id, "selected": discovery["devices"][0],
                    "backend": service.settings.backend, "notes": "Reusing active session."}
        raise DomainError("device_connected", "Cannot switch device while connected; close connection first.")
    selected: dict[str, Any] | None = None
    for dev in discovery.get("devices", []):
        if dev["id"] == device_id:
            selected = dev
            break

    if selected is None:
        raise DomainError(
            "device_not_found",
            f"Device '{device_id}' not found among discovered devices. No fallback permitted.",
        )

    settings: Settings = service.settings
    iface = selected["interface"]

    if iface in ("sim", "mujoco"):
        settings.backend = iface
    elif iface == "socketcan":
        settings.backend = "agx"
        settings.can_interface = "socketcan"
        settings.can_channel = selected["channel"]
        _set_cando_device_index(settings, None)
    elif iface == "agx_cando":
        dev_idx = selected.get("device_index")
        if dev_idx is None or dev_idx < 0:
            raise DomainError(
                "invalid_device",
                f"CANDO device '{device_id}' has invalid device index {dev_idx}.",
            )
        settings.backend = "agx"
        settings.can_interface = "agx_cando"
        settings.can_channel = "0"
        _set_cando_device_index(settings, int(dev_idx))
    else:
        raise DomainError("unknown_interface", f"Unknown interface '{iface}'")

    return {
        "device_id": device_id,
        "selected": selected,
        "backend": settings.backend,
        "notes": f"Selected device '{device_id}'.",
    }
