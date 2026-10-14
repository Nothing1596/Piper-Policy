"""Demonstration bundles: compile from human video, persist, retrieve.

Offline and host-side only: no hardware, no command authority, no live
execution claims. A stored demo is historical evidence; its outcome verdict
never certifies a current execution.
"""
from __future__ import annotations

from .compiler import CompileError, compile_demo
from .store import DemoStore, StoreError

__all__ = ["CompileError", "DemoStore", "StoreError", "compile_demo"]
