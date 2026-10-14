"""Local adapter lifecycle and per-ID receive-order wrapper.

Reset the owned CAN adapter channel before applying timing/start. This changes
the USB-CAN channel lifecycle, not robot state, and sends no CAN frames.
Discard timestamp regressions for each full CAN ID separately: unrelated
high-rate IDs must not starve slow driver-state IDs. This is ordering protection,
not a cross-message synchronization or device-to-host clock calibration.
"""
import ctypes
import threading
import time
from collections import Counter

from agx_cando.bus import (AgxCandoBus, BITRATE_TABLE, CandoBitTiming, CandoFrame,
                          CANDO_MODE_NORMAL, CANDO_MODE_LOOP_BACK, CANDO_MODE_NO_ECHO_BACK,
                          CANDO_ID_MASK, CANDO_ID_EXTENDED, CANDO_ID_RTR, CANDO_ID_ERR)
from can import BusABC, CanInitializationError, CanOperationError, Message


class SessionCandoBus(AgxCandoBus):
    _RX_JOIN_TIMEOUT_S = 1.0

    def __init__(self, *args, device_index=None, **kwargs):
        self._device_index = device_index
        self._close_lock = threading.RLock()
        self._cleanup_stopped = False
        self._cleanup_closed = False
        self._rx_timestamp_by_id = {}
        self.rx_order_discarded = Counter()
        self._stats_lock = threading.RLock()
        self._read_stats = Counter()
        self._rx_error = None
        self._last_native_read_s = self._last_native_frame_s = None
        self._last_device_timestamp_us = None
        self._last_can_error = None
        self.last_dequeued_received_s = None
        super().__init__(*args, **kwargs)

    def _recv_loop(self):
        # The vendor BOOL conflates timeout and I/O failure. Count it honestly;
        # do not invent an OS error from GetLastError after another DLL call.
        def read(timeout_ms):
            frame = CandoFrame()
            ok = self._api.dll.cando_frame_read(self._dev_handle, ctypes.byref(frame), timeout_ms)
            with self._stats_lock:
                self._last_native_read_s = time.monotonic()
                self._read_stats["read_calls"] += 1
                self._read_stats["frames" if ok else "no_data_or_error"] += 1
                if ok:
                    self._last_native_frame_s = self._last_native_read_s
                    self._last_device_timestamp_us = int(frame.timestamp_us)
            if ok:
                self._enqueue_frame(frame)
            return ok
        try:
            while not self._shutdown_flag.is_set():
                if not read(10):
                    self._shutdown_flag.wait(.001)  # Avoid a hot spin on immediate USB failures.
                    continue
                while not self._shutdown_flag.is_set() and read(0):
                    pass
        except Exception as exc:
            with self._stats_lock:
                self._rx_error = f"{type(exc).__name__}: {exc}"
            with self._queue_cond:
                self._queue_cond.notify_all()

    def diagnostics(self):
        now = time.monotonic()
        with self._stats_lock:
            result = dict(self._read_stats) | {
                "read_false_semantics": "timeout_or_native_error; DLL does not distinguish",
                "reader_error": self._rx_error,
                "reader_alive": bool(self._rx_thread and self._rx_thread.is_alive()),
                "last_read_age_s": None if self._last_native_read_s is None else now - self._last_native_read_s,
                "last_frame_age_s": None if self._last_native_frame_s is None else now - self._last_native_frame_s,
                "last_device_timestamp_us": self._last_device_timestamp_us,
                "last_can_error": self._last_can_error,
                "order_discarded": sum(self.rx_order_discarded.values())}
        with self._queue_cond:
            result["queue_depth"] = len(self._queue)
            result["oldest_queue_age_s"] = now - self._queue[0][4] if self._queue else None
        return result

    def _close_handles(self):
        # Also used on construction failure, before BusABC/transport attributes
        # necessarily exist. The owner retains this instance until cleanup succeeds.
        if not hasattr(self, "_close_lock"):
            self._close_lock = threading.RLock()
        with self._close_lock:
            flag = getattr(self, "_shutdown_flag", None)
            if flag is not None:
                flag.set()
            condition = getattr(self, "_queue_cond", None)
            if condition is not None:
                with condition:
                    condition.notify_all()
            thread = getattr(self, "_rx_thread", None)
            if thread is not None and thread.is_alive():
                if flag is None or thread is threading.current_thread():
                    raise CanOperationError("Cannot safely stop CAN receive thread; resources retained")
                thread.join(timeout=self._RX_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    raise CanOperationError("CAN receive thread did not exit; resources retained")

            def checked(name, handle):
                try:
                    succeeded = getattr(self._api.dll, name)(handle)
                except Exception as exc:
                    raise CanOperationError(f"{name} raised; resources retained") from exc
                if not succeeded:
                    raise CanOperationError(f"{name} failed; resources retained")

            handle = getattr(self, "_dev_handle", None)
            if handle:
                if not getattr(self, "_cleanup_stopped", False):
                    checked("cando_stop", handle)
                    self._cleanup_stopped = True
                if not getattr(self, "_cleanup_closed", False):
                    checked("cando_close", handle)
                    self._cleanup_closed = True
                checked("cando_free", handle)
                self._dev_handle = ctypes.c_void_p()
            handle = getattr(self, "_list_handle", None)
            if handle:
                checked("cando_list_free", handle)
                self._list_handle = ctypes.c_void_p()
            self._rx_thread = None

    def shutdown(self):
        if not hasattr(self, "_close_lock"):
            self._close_lock = threading.RLock()
        with self._close_lock:
            # Stop python-can send tasks before native handles can be freed.
            if hasattr(self, "_periodic_tasks"):
                self.stop_all_periodic_tasks()
            self._close_handles()
            if hasattr(self, "_periodic_tasks"):
                BusABC.shutdown(self)
            else:
                self._is_shutdown = True

    def _enqueue_frame(self, frame):
        key = (int(frame.channel), int(frame.can_id))
        timestamp = int(frame.timestamp_us)
        with self._stats_lock:
            previous = self._rx_timestamp_by_id.get(key)
            # Modulo subtraction handles 32-bit wrap for gaps below half the range.
            # Reopen after a device-clock reset; a reset must not silently rebase.
            if previous is not None and ((timestamp - previous) & 0xFFFFFFFF) >= 0x80000000:
                self.rx_order_discarded[key] += 1
                return
            self._rx_timestamp_by_id[key] = timestamp
        received = time.monotonic()
        record = (int(frame.can_id), int(frame.can_dlc), bytes(frame.data[:frame.can_dlc]),
                  timestamp, received, time.time())
        with self._stats_lock:
            if int(frame.can_dlc) > 8 or int(frame.channel) != 0:
                self._read_stats["malformed_frames"] += 1
                return
            if int(frame.can_id) & CANDO_ID_ERR:
                self._read_stats["can_error_frames"] += 1
                self._last_can_error = {"can_id": hex(int(frame.can_id)), "data": record[2].hex()}
        with self._queue_cond:
            # Keep bounded memory; report discarded backlog instead of silently
            # presenting arbitrarily old data with a new dequeue timestamp.
            if len(self._queue) >= 4096:
                self._queue.popleft()
                with self._stats_lock:
                    self._read_stats["queue_overflow_dropped"] += 1
            self._queue.append(record)
            self._queue_cond.notify()

    def _recv_internal(self, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._queue_cond:
            while not self._queue:
                if self._rx_error:
                    raise CanOperationError(self._rx_error)
                if self._shutdown_flag.is_set():
                    return None, False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None, False
                self._queue_cond.wait(remaining)
            if self._rx_error:
                raise CanOperationError(self._rx_error)
            ident, dlc, data, device_stamp, received, wall = self._queue.popleft()
        self.last_dequeued_received_s = received
        return Message(arbitration_id=ident & CANDO_ID_MASK, data=data, dlc=dlc,
                       is_extended_id=bool(ident & CANDO_ID_EXTENDED),
                       is_remote_frame=bool(ident & CANDO_ID_RTR), is_error_frame=bool(ident & CANDO_ID_ERR),
                       channel=self._channel_index, timestamp=wall), False

    def _open_device(self):
        dll = self._api.dll
        if self._bitrate not in BITRATE_TABLE:
            raise CanInitializationError('Unsupported bitrate')
        try:
            if not dll.cando_list_malloc(ctypes.byref(self._list_handle)):
                raise CanInitializationError('cando_list_malloc failed')
            if not dll.cando_list_scan(self._list_handle):
                raise CanInitializationError('cando_list_scan failed')
            count = ctypes.c_uint8()
            if not dll.cando_list_num(self._list_handle, ctypes.byref(count)):
                raise CanInitializationError('cando_list_num failed')
            index = getattr(self, '_device_index', None)
            if self._channel_index != 0:
                raise CanInitializationError('CANDO CAN channel must be 0')
            if index is None:
                if count.value != 1:
                    raise CanInitializationError('Select an adapter explicitly when multiple devices are connected')
                index = 0
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count.value:
                raise CanInitializationError('Selected CANDO adapter is no longer present')
            if not dll.cando_malloc(self._list_handle, index, ctypes.byref(self._dev_handle)):
                raise CanInitializationError('cando_malloc failed')
            # One native open attempt, rather than the upstream 20 x 5s retry.
            if not dll.cando_open(self._dev_handle):
                raise CanInitializationError('cando_open failed')
            if not dll.cando_stop(self._dev_handle):
                raise CanInitializationError('Adapter channel stop-before-start failed')
            timing = CandoBitTiming(*BITRATE_TABLE[self._bitrate])
            if not dll.cando_set_timing(self._dev_handle, ctypes.byref(timing)):
                raise CanInitializationError('cando_set_timing failed')
            mode = CANDO_MODE_LOOP_BACK if self._loopback else CANDO_MODE_NORMAL
            if not self._receive_own_messages and not self._loopback:
                mode |= CANDO_MODE_NO_ECHO_BACK
            if not dll.cando_start(self._dev_handle, mode):
                raise CanInitializationError('cando_start failed')
            self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._rx_thread.start()
        except Exception:
            self._close_handles()
            raise
