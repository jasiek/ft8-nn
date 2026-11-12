#!/usr/bin/env python3
"""Compare ft8-nn with an OpenWebRX FT8 decoder on the same live audio.

The receiver picker supports arrow keys and live substring filtering.  After a
receiver/band pair is selected, this client tunes its OpenWebRX profile to the
standard FT8 dial frequency, receives the server's 12 kHz audio and FT8
decodes, decodes each UTC slot with libft8nn, and displays the results in two
frequency-ordered columns.

Build the C library first if needed:

    make -C c
    python3 tools/live_compare.py

Press q while comparing to quit.
"""

from __future__ import annotations

import argparse
import array
import base64
import collections
import ctypes
import hashlib
import json
import os
import queue
import re
import shutil
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import curses
except ImportError:  # Not included with the standard CPython distribution on Windows.
    curses = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "datasets" / "receiverbook-directory.json"
DEFAULT_MODEL = ROOT / "models" / "v3" / "ft8nn.onnx"
FT8_FREQUENCIES = {
    "2190m": 136_000,
    "630m": 474_200,
    "160m": 1_840_000,
    "80m": 3_573_000,
    "60m": 5_357_000,
    "40m": 7_074_000,
    "30m": 10_136_000,
    "20m": 14_074_000,
    "17m": 18_100_000,
    "15m": 21_074_000,
    "12m": 24_915_000,
    "10m": 28_074_000,
    "8m": 40_680_000,
    "6m": 50_313_000,
    "4m": 70_154_000,
    "2m": 144_174_000,
    "70cm": 432_174_000,
    "23cm": 1_296_174_000,
    "13cm": 2_320_174_000,
    "6cm": 5_760_174_000,
    "3cm": 10_368_174_000,
}
METHODS = ("bp", "osd", "bp_ap", "osd_ap", "bp_er", "osd_er", "osd_bp", "bp_cx", "osd_cx", "rp")


@dataclass(frozen=True)
class Decode:
    freq: float
    dt: float
    snr: float
    text: str
    method: str = ""


def normalize_message(text: str) -> str:
    text = " ".join(text.strip().upper().split())
    return re.sub(r"<[^>]*>", "<...>", text)


def match_decodes(
    ours: list[Decode], remote: list[Decode]
) -> list[tuple[Decode | None, Decode | None]]:
    """Pair identical messages, closest in frequency first; order all rows by frequency."""
    remaining = list(remote)
    rows: list[tuple[Decode | None, Decode | None]] = []
    for local in sorted(ours, key=lambda d: d.freq):
        candidates = [
            (abs(local.freq - candidate.freq), index)
            for index, candidate in enumerate(remaining)
            if normalize_message(candidate.text) == normalize_message(local.text)
        ]
        if candidates:
            _, index = min(candidates)
            rows.append((local, remaining.pop(index)))
        else:
            rows.append((local, None))
    rows.extend((None, candidate) for candidate in remaining)
    return sorted(
        rows,
        key=lambda pair: (
            sum(d.freq for d in pair if d is not None) / sum(d is not None for d in pair),
            normalize_message(next(d.text for d in pair if d is not None)),
        ),
    )


def filter_entries(entries: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    words = query.casefold().split()

    def searchable(entry: dict[str, Any]) -> str:
        receiver = entry["receiver"]
        return " ".join(
            str(value)
            for value in (
                entry.get("band", ""),
                receiver.get("label", ""),
                receiver.get("site", ""),
                receiver.get("url", ""),
            )
        ).casefold()

    return [entry for entry in entries if all(word in searchable(entry) for word in words)]


def entry_label(entry: dict[str, Any]) -> str:
    receiver = entry["receiver"]
    rate = entry.get("ft8_decodes_per_minute")
    traffic = f"  {rate:,.2f}/min" if isinstance(rate, (int, float)) else ""
    return (
        f"{entry['band']:>5}{traffic}  "
        f"{receiver.get('label') or receiver.get('site') or receiver['url']}  [{receiver['url']}]"
    )


def hide_cursor() -> None:
    if curses is None:
        return
    try:
        curses.curs_set(0)
    except curses.error:
        pass


def pick_receiver(screen: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if curses is None:
        raise RuntimeError("the receiver picker requires Python curses")
    hide_cursor()
    screen.keypad(True)
    query = ""
    selected = 0
    offset = 0
    while True:
        matches = filter_entries(entries, query)
        if matches:
            selected = min(selected, len(matches) - 1)
        else:
            selected = 0
        height, width = screen.getmaxyx()
        visible = max(1, height - 4)
        if selected < offset:
            offset = selected
        elif selected >= offset + visible:
            offset = selected - visible + 1

        screen.erase()
        screen.addnstr(0, 0, "Select receiver/band — type to filter, ↑/↓ to move, Enter to connect, Esc to quit", width - 1)
        screen.addnstr(1, 0, f"Filter: {query}", width - 1, curses.A_BOLD)
        if not matches:
            screen.addnstr(3, 0, "No matches", width - 1)
        for row, entry in enumerate(matches[offset : offset + visible], start=3):
            index = offset + row - 3
            attr = curses.A_REVERSE if index == selected else curses.A_NORMAL
            screen.addnstr(row, 0, entry_label(entry), width - 1, attr)
        screen.refresh()

        key = screen.get_wch()
        if key in (curses.KEY_UP,):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN,):
            selected = min(max(0, len(matches) - 1), selected + 1)
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - visible)
        elif key == curses.KEY_NPAGE:
            selected = min(max(0, len(matches) - 1), selected + visible)
        elif key in ("\n", "\r", curses.KEY_ENTER):
            if matches:
                return matches[selected]
        elif key == "\x1b":
            return None
        elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
            query = query[:-1]
            selected = offset = 0
        elif isinstance(key, str) and key.isprintable():
            query += key
            selected = offset = 0


class WebSocket:
    """Small RFC 6455 client sufficient for OpenWebRX (no extensions)."""

    def __init__(self, url: str, timeout: float = 10.0):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError(f"invalid WebSocket URL: {url}")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        sock = socket.create_connection((parsed.hostname, port), timeout=timeout)
        if parsed.scheme == "wss":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parsed.hostname)
        sock.settimeout(timeout)
        self.sock = sock
        self.buffer = bytearray()
        self.send_lock = threading.Lock()
        self.fragments = bytearray()
        self.fragment_opcode: int | None = None

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
        host = parsed.hostname if port in (80, 443) else f"{parsed.hostname}:{port}"
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            f"Origin: {'https' if parsed.scheme == 'wss' else 'http'}://{host}\r\n"
            "User-Agent: ft8-nn-live-compare/1\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = self._read_until(b"\r\n\r\n")
        header, leftover = response.split(b"\r\n\r\n", 1)
        self.buffer.extend(leftover)
        lines = header.decode("iso-8859-1").split("\r\n")
        if not lines[0].startswith("HTTP/1.1 101"):
            raise ConnectionError(f"WebSocket upgrade failed: {lines[0]}")
        headers = {k.strip().lower(): v.strip() for k, v in (line.split(":", 1) for line in lines[1:] if ":" in line)}
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if headers.get("sec-websocket-accept") != expected:
            raise ConnectionError("invalid Sec-WebSocket-Accept")
        sock.settimeout(None)

    def _read_until(self, marker: bytes) -> bytes:
        data = bytearray()
        while marker not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise EOFError("connection closed during WebSocket upgrade")
            data.extend(chunk)
            if len(data) > 64 * 1024:
                raise ConnectionError("oversized WebSocket response headers")
        return bytes(data)

    def _read_exact(self, size: int) -> bytes:
        while len(self.buffer) < size:
            chunk = self.sock.recv(max(4096, size - len(self.buffer)))
            if not chunk:
                raise EOFError("WebSocket closed")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def _send_frame(self, opcode: int, payload: bytes = b"") -> None:
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", first, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first, 0x80 | 127, length)
        mask = os.urandom(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        with self.send_lock:
            self.sock.sendall(header + mask + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(1, text.encode("utf-8"))

    def send_json(self, value: Any) -> None:
        self.send_text(json.dumps(value, separators=(",", ":")))

    def recv(self) -> tuple[str, bytes | str]:
        while True:
            first, second = self._read_exact(2)
            final, opcode = bool(first & 0x80), first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 8:
                raise EOFError("WebSocket closed by server")
            if opcode == 9:
                self._send_frame(10, payload)
                continue
            if opcode == 10:
                continue
            if opcode in (1, 2):
                self.fragment_opcode = opcode
                self.fragments = bytearray(payload)
            elif opcode == 0 and self.fragment_opcode is not None:
                self.fragments.extend(payload)
            else:
                continue
            if not final:
                continue
            opcode = self.fragment_opcode
            payload = bytes(self.fragments)
            self.fragment_opcode = None
            self.fragments.clear()
            return ("text", payload.decode("utf-8")) if opcode == 1 else ("binary", payload)

    def close(self) -> None:
        try:
            self._send_frame(8, struct.pack("!H", 1000))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


IMA_INDEX = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)
IMA_STEP = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253,
    279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166,
    1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026,
    4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635,
    13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
)


class ImaAdpcmDecoder:
    def __init__(self) -> None:
        self.step_index = 0
        self.predictor = 0
        self.step = 0
        self.phase = 0
        self.sync_match = 0
        self.sync_data = bytearray()
        self.sync_counter = 0

    def _nibble(self, nibble: int) -> int:
        self.step_index = min(88, max(0, self.step_index + IMA_INDEX[nibble]))
        diff = self.step >> 3
        if nibble & 1:
            diff += self.step >> 2
        if nibble & 2:
            diff += self.step >> 1
        if nibble & 4:
            diff += self.step
        if nibble & 8:
            diff = -diff
        self.predictor = min(32767, max(-32768, self.predictor + diff))
        self.step = IMA_STEP[self.step_index]
        return self.predictor

    def decode(self, data: bytes) -> array.array:
        output = array.array("h")
        sync = b"SYNC"
        for value in data:
            if self.phase == 0:
                if value == sync[self.sync_match]:
                    self.sync_match += 1
                else:
                    self.sync_match = 1 if value == sync[0] else 0
                if self.sync_match == 4:
                    self.sync_data.clear()
                    self.phase = 1
            elif self.phase == 1:
                self.sync_data.append(value)
                if len(self.sync_data) == 4:
                    self.step_index, self.predictor = struct.unpack("<hh", self.sync_data)
                    self.step_index = min(88, max(0, self.step_index))
                    self.sync_counter = 1000
                    self.phase = 2
            else:
                output.append(self._nibble(value & 0x0F))
                output.append(self._nibble(value >> 4))
                if self.sync_counter == 0:
                    self.sync_match = 0
                    self.phase = 0
                else:
                    self.sync_counter -= 1
        return output


def pcm16(data: bytes) -> array.array:
    result = array.array("h")
    result.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder != "little":
        result.byteswap()
    return result


def parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000 if value > 10_000_000_000 else float(value)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def audio_frequency(value: Any, dial: int) -> float:
    """Normalize OpenWebRX's absolute RF frequency (or legacy audio Hz) to audio Hz."""
    frequency = float(value)
    return frequency - dial if frequency > 100_000 else frequency


def resolve_receiver_url(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ft8-nn-live-compare/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.geturl()


def metrics_url(receiver_url: str) -> str:
    """Return the metrics endpoint while preserving an OpenWebRX subpath."""
    return urllib.parse.urljoin(receiver_url.rstrip("/") + "/", "metrics.json")


class OpenWebRXClient:
    def __init__(
        self,
        url: str,
        band: str,
        on_audio: Callable[[array.array, float], None],
        on_decode: Callable[[int, Decode], None],
        on_status: Callable[[str], None],
        timeout: float = 10.0,
    ) -> None:
        self.url = url
        self.band = band
        self.dial = FT8_FREQUENCIES[band]
        self.on_audio = on_audio
        self.on_decode = on_decode
        self.on_status = on_status
        self.timeout = timeout
        self.ws: WebSocket | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.profiles: list[dict[str, str]] = []
        self.modes: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}
        self.profile_generation = 0
        self.error: Exception | None = None
        self.audio_compression = "none"
        self.adpcm = ImaAdpcmDecoder()

    def start(self) -> None:
        final_url = resolve_receiver_url(self.url, self.timeout)
        parsed = urllib.parse.urlparse(final_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path if parsed.path.endswith("/") else parsed.path.rsplit("/", 1)[0] + "/"
        ws_path = urllib.parse.urljoin(base_path, "ws/")
        ws_url = urllib.parse.urlunparse((scheme, parsed.netloc, ws_path, "", "", ""))
        self.on_status(f"connecting to {ws_url}")
        self.ws = WebSocket(ws_url, self.timeout)
        self.ws.send_text("SERVER DE CLIENT client=ft8-nn-live-compare type=receiver")
        self.ws.send_json({"type": "connectionproperties", "params": {"output_rate": 12000, "hd_output_rate": 0}})
        self.thread = threading.Thread(target=self._read_loop, name="openwebrx", daemon=True)
        self.thread.start()
        self._configure()

    def _current_profile(self) -> str | None:
        if self.config.get("sdr_id") is None or self.config.get("profile_id") is None:
            return None
        return f"{self.config['sdr_id']}|{self.config['profile_id']}"

    def _covers_dial(self) -> bool:
        center = self.config.get("center_freq")
        rate = self.config.get("samp_rate")
        return bool(center and rate and center - rate / 2 <= self.dial and self.dial + 3000 <= center + rate / 2)

    def _wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> None:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        with self.condition:
            while not predicate():
                if self.error:
                    raise self.error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for OpenWebRX configuration")
                self.condition.wait(remaining)

    def _configure(self) -> None:
        deadline = time.monotonic() + self.timeout
        self._wait_for(
            lambda: bool(self.profiles and self.modes and self.config.get("center_freq") and self.config.get("samp_rate")),
            max(0, deadline - time.monotonic()),
        )
        if not any(mode.get("modulation") == "ft8" for mode in self.modes):
            raise RuntimeError("receiver does not advertise the FT8 decoder")

        band_pattern = re.compile(rf"(?<!\d){re.escape(self.band)}(?!\w)", re.IGNORECASE)
        profiles = sorted(
            self.profiles,
            key=lambda profile: (
                0 if profile["id"] == self._current_profile() else 1,
                0 if band_pattern.search(profile.get("name", "")) else 1,
                0 if "ft8" in profile.get("name", "").casefold() else 1,
                profile.get("name", ""),
            ),
        )
        selected = None
        for profile in profiles:
            profile_id = profile["id"]
            if profile_id != self._current_profile():
                generation = self.profile_generation
                self.ws.send_json({"type": "selectprofile", "params": {"profile": profile_id}})
                self._wait_for(
                    lambda: self._current_profile() == profile_id and self.profile_generation > generation,
                    max(0, deadline - time.monotonic()),
                )
            if self._covers_dial():
                selected = profile
                break
        if selected is None:
            raise RuntimeError(f"no available receiver profile covers {self.band} FT8 at {self.dial / 1e6:.6f} MHz")

        mode = next(mode for mode in self.modes if mode.get("modulation") == "ft8")
        bandpass = mode.get("bandpass") or {"low_cut": 0, "high_cut": 3000}
        underlying = (mode.get("underlying") or ["usb"])[0]
        params = {
            "low_cut": bandpass["low_cut"],
            "high_cut": bandpass["high_cut"],
            "offset_freq": round(self.dial - self.config["center_freq"]),
            "mod": underlying,
            "squelch_level": -150,
            "secondary_mod": "ft8",
        }
        self.ws.send_json({"type": "dspcontrol", "params": params})
        self.ws.send_json({"type": "dspcontrol", "action": "start"})
        self.on_status(f"streaming {selected['name']} at {self.dial / 1e6:.6f} MHz")

    def _read_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                kind, payload = self.ws.recv()
                if kind == "binary":
                    data = payload
                    if data and data[0] == 2:
                        samples = self.adpcm.decode(data[1:]) if self.audio_compression == "adpcm" else pcm16(data[1:])
                        if samples:
                            self.on_audio(samples, time.time())
                    continue
                text = payload
                if text.startswith("CLIENT DE SERVER"):
                    continue
                message = json.loads(text)
                msg_type = message.get("type")
                value = message.get("value")
                with self.condition:
                    if msg_type == "config" and isinstance(value, dict):
                        self.config.update(value)
                        self.audio_compression = self.config.get("audio_compression", self.audio_compression)
                        if "profile_id" in value or "sdr_id" in value:
                            self.profile_generation += 1
                    elif msg_type == "profiles" and isinstance(value, list):
                        self.profiles = value
                    elif msg_type == "modes" and isinstance(value, list):
                        self.modes = value
                    elif msg_type == "backoff":
                        raise RuntimeError(f"receiver busy: {message.get('reason', 'backoff')}")
                    elif msg_type in ("sdr_error", "demodulator_error"):
                        raise RuntimeError(str(value))
                    self.condition.notify_all()
                if msg_type == "secondary_demod" and isinstance(value, dict) and str(value.get("mode", "")).upper() == "FT8":
                    slot = int(parse_timestamp(value["timestamp"]) // 15 * 15)
                    frequency = audio_frequency(value.get("freq", 0), self.dial)
                    # OpenWebRX WSJT messages normally carry absolute RF Hz, while libft8nn
                    # reports the tone-zero audio offset.  Older variants already send the
                    # offset, so only translate values that cannot be audio frequencies.
                    self.on_decode(
                        slot,
                        Decode(
                            frequency,
                            float(value.get("dt", 0)),
                            float(value.get("db", 0)),
                            str(value.get("msg", "")),
                            "remote",
                        ),
                    )
        except Exception as error:
            if not self.stop_event.is_set():
                self.error = error
                self.on_status(f"OpenWebRX error: {error}")
                with self.condition:
                    self.condition.notify_all()

    def close(self) -> None:
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)


class CDecode(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char * 48),
        ("f0", ctypes.c_double),
        ("dt", ctypes.c_double),
        ("snr", ctypes.c_double),
        ("bits77", ctypes.c_uint8 * 10),
        ("method", ctypes.c_int),
        ("metric", ctypes.c_int),
        ("pass_", ctypes.c_int),
    ]


def default_library() -> Path:
    names = {"darwin": "libft8nn.dylib", "win32": "ft8nn.dll"}
    return ROOT / "c" / "build" / names.get(sys.platform, "libft8nn.so")


class LocalDecoder:
    def __init__(self, library: Path, model: Path, threads: int = 0) -> None:
        if not library.exists():
            raise FileNotFoundError(f"{library} not found; run: make -C c")
        if not model.exists():
            raise FileNotFoundError(model)
        self.lib = ctypes.CDLL(str(library))
        self.lib.ft8nn_open.argtypes = (ctypes.c_char_p, ctypes.c_int)
        self.lib.ft8nn_open.restype = ctypes.c_void_p
        self.lib.ft8nn_decode_at.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.POINTER(CDecode),
            ctypes.c_int,
        )
        self.lib.ft8nn_decode_at.restype = ctypes.c_int
        self.lib.ft8nn_close.argtypes = (ctypes.c_void_p,)
        self.ctx = self.lib.ft8nn_open(os.fsencode(model), threads)
        if not self.ctx:
            raise RuntimeError("ft8nn_open failed")

    def decode(self, slot: int, pcm: array.array) -> list[Decode]:
        samples = array.array("f", pcm)
        pointer = (ctypes.c_float * len(samples)).from_buffer(samples)
        output = (CDecode * 256)()
        count = self.lib.ft8nn_decode_at(self.ctx, pointer, len(samples), slot, output, len(output))
        if count < 0:
            raise RuntimeError("ft8nn_decode_at failed")
        return [
            Decode(
                item.f0,
                item.dt,
                item.snr,
                bytes(item.text).split(b"\0", 1)[0].decode("utf-8", "replace"),
                METHODS[item.method] if 0 <= item.method < len(METHODS) else "?",
            )
            for item in output[: min(count, len(output))]
        ]

    def close(self) -> None:
        if self.ctx:
            self.lib.ft8nn_close(self.ctx)
            self.ctx = None


class SlotAssembler:
    def __init__(self, submit: Callable[[int, array.array], None], delay: float = 0.0, rate: int = 12000) -> None:
        self.submit = submit
        self.delay = delay
        self.rate = rate
        self.slot: int | None = None
        self.start_at: float | None = None
        self.samples = array.array("h")
        self.lock = threading.Lock()

    def push(self, samples: array.array, received_at: float) -> None:
        with self.lock:
            if self.slot is None:
                self.slot = (int(received_at) // 15 + 1) * 15
                self.start_at = self.slot + self.delay
            chunk_start = received_at - len(samples) / self.rate
            if self.start_at is not None and chunk_start + len(samples) / self.rate <= self.start_at:
                return
            if self.start_at is not None:
                skip = max(0, min(len(samples), round((self.start_at - chunk_start) * self.rate)))
                samples = samples[skip:]
                self.start_at = None
            self.samples.extend(samples)
            slot_size = 15 * self.rate
            while len(self.samples) >= slot_size:
                complete = self.samples[:slot_size]
                del self.samples[:slot_size]
                slot = self.slot
                self.slot += 15
                self.submit(slot, complete)


class ComparisonState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ours: dict[int, list[Decode]] = collections.defaultdict(list)
        self.remote: dict[int, list[Decode]] = collections.defaultdict(list)
        self.status = "starting"
        self.latest_audio_slot: int | None = None
        self.decode_seconds: dict[int, float] = {}
        self.error: str | None = None

    def trim(self) -> None:
        slots = sorted(set(self.ours) | set(self.remote))
        for old in slots[:-8]:
            self.ours.pop(old, None)
            self.remote.pop(old, None)
            self.decode_seconds.pop(old, None)


def format_decode(decode: Decode | None, width: int) -> str:
    if decode is None:
        return ""
    prefix = f"{decode.freq:6.1f} {decode.snr:+4.0f} {decode.dt:+5.2f} "
    return (prefix + decode.text)[:width]


def draw_comparison(screen: Any, entry: dict[str, Any], state: ComparisonState) -> None:
    screen.erase()
    height, width = screen.getmaxyx()
    receiver = entry["receiver"]
    title = f"{receiver.get('label', receiver['url'])} — {entry['band']} FT8 — q to quit"
    screen.addnstr(0, 0, title, width - 1, curses.A_BOLD if curses is not None else 0)
    with state.lock:
        status = state.status
        slots = sorted(set(state.ours) | set(state.remote))
        slot = slots[-1] if slots else None
        ours = list(state.ours.get(slot, [])) if slot is not None else []
        remote = list(state.remote.get(slot, [])) if slot is not None else []
        elapsed = state.decode_seconds.get(slot) if slot is not None else None
        audio_slot = state.latest_audio_slot
    screen.addnstr(1, 0, status, width - 1)
    screen.addnstr(2, 0, f"Metrics: {metrics_url(receiver['url'])}", width - 1)
    rate = entry.get("ft8_decodes_per_minute")
    rate_text = f"{rate:,.2f}/min" if isinstance(rate, (int, float)) else "unknown"
    decode_count = entry.get("ft8_decodes_total", entry.get("ft8_decodes"))
    count_text = f"{decode_count:,}" if isinstance(decode_count, int) else "unknown"
    screen.addnstr(
        3, 0,
        f"FT8 traffic at directory build: {rate_text}  total since restart: {count_text}",
        width - 1,
    )
    if slot is None:
        waiting = "Waiting for the next UTC slot and remote decodes…"
        screen.addnstr(5, 0, waiting, width - 1)
        screen.refresh()
        return
    stamp = datetime.fromtimestamp(slot, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    detail = f"Slot {stamp}  local {len(ours)}  remote {len(remote)}"
    if elapsed is not None:
        detail += f"  local decode {elapsed:.1f}s"
    if audio_slot is not None and audio_slot > slot:
        detail += "  (local decode pending for newer audio)"
    screen.addnstr(4, 0, detail, width - 1)

    gap = 3
    col = max(20, (width - gap) // 2)
    right = col + gap
    underline = curses.A_UNDERLINE if curses is not None else 0
    screen.addnstr(6, 0, "OUR DECODER  Hz  dB   DT  message", col, underline)
    if right < width:
        screen.addnstr(6, right, "REMOTE DECODER  Hz  dB   DT  message", max(1, width - right - 1), underline)
    rows = match_decodes(ours, remote)
    for row, (local, server) in enumerate(rows[: max(0, height - 8)], start=7):
        screen.addnstr(row, 0, format_decode(local, col), col)
        if right < width:
            screen.addnstr(row, right, format_decode(server, width - right - 1), max(1, width - right - 1))
    screen.refresh()


def run_comparison(screen: Any, entry: dict[str, Any], args: argparse.Namespace) -> None:
    hide_cursor()
    screen.nodelay(True)
    screen.keypad(True)
    state = ComparisonState()
    jobs: queue.Queue[tuple[int, array.array] | None] = queue.Queue(maxsize=2)
    decode_stop = threading.Event()
    decoder = LocalDecoder(args.library, args.model, args.threads)

    def status(text: str) -> None:
        with state.lock:
            state.status = text

    def add_remote(slot: int, decode: Decode) -> None:
        with state.lock:
            state.remote[slot].append(decode)
            state.trim()

    def submit(slot: int, samples: array.array) -> None:
        with state.lock:
            state.latest_audio_slot = slot
        try:
            jobs.put_nowait((slot, samples))
        except queue.Full:
            status("local decoder fell behind; dropping one slot")

    assembler = SlotAssembler(submit, args.audio_delay)

    def decode_loop() -> None:
        while not decode_stop.is_set():
            try:
                job = jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None or decode_stop.is_set():
                break
            slot, samples = job
            started = time.monotonic()
            try:
                result = decoder.decode(slot, samples)
                with state.lock:
                    state.ours[slot] = result
                    state.decode_seconds[slot] = time.monotonic() - started
                    state.trim()
            except Exception as error:
                with state.lock:
                    state.error = str(error)
                    state.status = f"local decoder error: {error}"

    worker = threading.Thread(target=decode_loop, name="ft8nn", daemon=True)
    worker.start()
    client = OpenWebRXClient(entry["receiver"]["url"], entry["band"], assembler.push, add_remote, status, args.timeout)
    try:
        try:
            client.start()
        except Exception as error:
            status(f"connection failed: {error} — press q to quit")
        while True:
            draw_comparison(screen, entry, state)
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                break
            time.sleep(0.1)
    finally:
        client.close()
        decode_stop.set()
        try:
            jobs.put_nowait(None)
        except queue.Full:
            pass
        worker.join()
        decoder.close()


def fallback_picker(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    query = input("Receiver substring (blank for all): ")
    matches = filter_entries(entries, query)
    for index, entry in enumerate(matches[:100], 1):
        print(f"{index:3d}. {entry_label(entry)}")
    if not matches:
        return None
    selected = int(input("Number: ")) - 1
    return matches[selected] if 0 <= selected < len(matches) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--library", type=Path, default=default_library())
    parser.add_argument("--threads", type=int, default=0, help="local decoder threads; 0 uses performance cores")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--audio-delay", type=float, default=0.0, help="seconds from UTC boundary until its audio arrives")
    args = parser.parse_args(argv)
    document = json.loads(args.directory.read_text())
    entries = [entry for entry in document["entries"] if entry.get("band") in FT8_FREQUENCIES]
    if not entries:
        parser.error("receiver directory contains no supported entries")
    if curses is None:
        print("The live comparison terminal UI requires Python curses, which is not available.", file=sys.stderr)
        return 2
    if not sys.stdin.isatty() or not sys.stdout.isatty() or not os.environ.get("TERM"):
        entry = fallback_picker(entries)
        if entry is None:
            return 1
        print("The live comparison requires an interactive terminal.", file=sys.stderr)
        return 2

    selected: dict[str, Any] | None = None

    def ui(screen: Any) -> None:
        nonlocal selected
        selected = pick_receiver(screen, entries)
        if selected is not None:
            run_comparison(screen, selected, args)

    curses.wrapper(ui)
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
