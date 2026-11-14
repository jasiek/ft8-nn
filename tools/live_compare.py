#!/usr/bin/env python3
"""Compare ft8-nn with an UberSDR FT8 decoder on the same live audio.

The receiver picker supports arrow keys and live substring filtering.  After a
receiver/band pair is selected, this client tunes its UberSDR stream to the
standard FT8 dial frequency, receives the server's 12 kHz audio and FT8
decodes, decodes each UTC slot with libft8nn, and displays the results alongside
band-filtered PSK Reporter reception reports in three columns.

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
import ctypes.util
import hashlib
import json
import math
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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import curses
except ImportError:  # Not included with the standard CPython distribution on Windows.
    curses = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = "https://instances.ubersdr.org/api/instances?conditions=true"
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
PSK_CONTEXT_SECONDS = 5 * 60
SLOT_HISTORY = 40  # Ten minutes of 15-second slots for delayed reception reports.
METHODS = ("bp", "osd", "bp_ap", "osd_ap", "bp_er", "osd_er", "osd_bp", "bp_cx", "osd_cx", "rp")


@dataclass(frozen=True)
class Decode:
    freq: float
    dt: float
    snr: float
    text: str
    method: str = ""


@dataclass
class PskSignal:
    call: str
    grid: str
    freq: float
    reports: int = 0
    slot: int | None = None


def parse_psk_report(value: Any, band: str) -> tuple[int, PskSignal] | None:
    """Use transmission time when supplied; raw receiver time is the fallback."""
    if not isinstance(value, dict) or value.get("md") != "FT8" or value.get("b") != band:
        return None
    try:
        stamp = float(value.get("t_tx") if value.get("t_tx") is not None else value["t"])
        freq = float(value["f"])
        call = value["sc"].strip().upper()
        if not call or not math.isfinite(stamp) or stamp < 0 or not math.isfinite(freq) or freq <= 0:
            return None
        grid = str(value.get("sl", ""))
        slot = int(stamp // 15) * 15
        return slot, PskSignal(call, grid, freq, 1, slot)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return None


class PskReporterClient:
    """Small MQTT 3.1.1 QoS-0 subscriber over TLS, with reconnect and keepalive."""

    def __init__(self, band: str, on_report: Callable[[Any], None],
                 on_status: Callable[[str], None], timeout: float = 10.0):
        self.topic = f"pskr/filter/v2/{band}/FT8/#"
        self.on_report = on_report
        self.on_status = on_status
        self.timeout = timeout
        self.stop = threading.Event()
        self.sock: socket.socket | None = None
        self.thread = threading.Thread(target=self._run, name="pskreporter", daemon=True)

    @staticmethod
    def _packet(header: int, payload: bytes) -> bytes:
        size = len(payload)
        encoded = bytearray([header])
        while True:
            size, digit = divmod(size, 128)
            encoded.append(digit | (128 if size else 0))
            if not size:
                return bytes(encoded) + payload

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self.on_status("connecting")
                raw = socket.create_connection(("mqtt.pskreporter.info", 1884), self.timeout)
                self.sock = raw
                try:
                    self.sock = ssl.create_default_context().wrap_socket(
                        raw, server_hostname="mqtt.pskreporter.info")
                except Exception:
                    raw.close()
                    raise
                self.sock.settimeout(1.0)
                client_id = ("ft8nn-" + uuid.uuid4().hex[:16]).encode()
                connect = b"\x00\x04MQTT\x04\x02\x00\x3c" + struct.pack("!H", len(client_id)) + client_id
                self.sock.sendall(self._packet(0x10, connect))
                buffer = bytearray()
                connected = False
                subscribed = False
                last_ping = time.monotonic()
                deadline = last_ping + self.timeout
                ping_pending = False
                while not self.stop.is_set():
                    now = time.monotonic()
                    if not subscribed and now > deadline:
                        raise TimeoutError("MQTT handshake timed out")
                    if connected and now - last_ping >= 30:
                        if ping_pending:
                            raise TimeoutError("MQTT keepalive timed out")
                        self.sock.sendall(b"\xc0\x00")
                        last_ping = now
                        ping_pending = True
                    try:
                        data = self.sock.recv(65536)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("MQTT connection closed")
                    buffer.extend(data)
                    while len(buffer) >= 2:
                        size, multiplier, offset = 0, 1, 1
                        while True:
                            if offset >= len(buffer):
                                break
                            digit = buffer[offset]
                            offset += 1
                            size += (digit & 127) * multiplier
                            if not digit & 128:
                                break
                            if offset >= 5:
                                raise ValueError("invalid MQTT packet length")
                            multiplier *= 128
                        if buffer[offset - 1] & 128:
                            break
                        if size > 1_048_576:
                            raise ValueError("MQTT packet too large")
                        if len(buffer) < offset + size:
                            break
                        header = buffer[0]
                        payload = bytes(buffer[offset:offset + size])
                        del buffer[:offset + size]
                        kind = header >> 4
                        if kind == 2:
                            if payload != b"\x00\x00":
                                raise ConnectionError("MQTT connection refused")
                            connected = True
                            topic = self.topic.encode()
                            self.sock.sendall(self._packet(0x82, b"\x00\x01" + struct.pack("!H", len(topic)) + topic + b"\x00"))
                        elif kind == 9:
                            if payload != b"\x00\x01\x00":
                                raise ConnectionError("MQTT subscription refused")
                            subscribed = True
                            self.on_status("connected")
                        elif kind == 13:
                            ping_pending = False
                        elif kind == 3 and subscribed and len(payload) >= 2:
                            # Subscription requests QoS 0, so no packet identifier follows the topic.
                            topic_size = int.from_bytes(payload[:2], "big")
                            if (header >> 1) & 3 or len(payload) < 2 + topic_size:
                                continue
                            try:
                                self.on_report(json.loads(payload[2 + topic_size:]))
                            except (ValueError, UnicodeError):
                                continue
            except Exception as error:
                if not self.stop.is_set():
                    self.on_status(f"{error}; retrying")
            finally:
                if self.sock is not None:
                    self.sock.close()
                    self.sock = None
            self.stop.wait(3)

    def close(self) -> None:
        self.stop.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=self.timeout + 2)


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


def transmitting_call(text: str) -> str | None:
    """Extract the sender, never the addressed station, from structured FT8 text."""
    fields = text.upper().split()
    if not fields:
        return None
    if fields[0] == "TU;":  # RTTY Roundup prefix
        fields = fields[1:]
    elif ";" in text:
        # DXpedition: recipient RR73; recipient <sender> report
        fields = text.upper().split(";", 1)[1].split()
    if len(fields) < 2:
        return None

    def call(token: str) -> str | None:
        token = token.strip("<>")
        if (re.fullmatch(r"[A-Z0-9/]+", token)
                and re.search(r"[A-Z]", token) and re.search(r"[0-9]", token)
                and not re.fullmatch(r"[A-R]{2}[0-9]{2}(?:[A-X]{2})?", token)
                and token not in {"RR73", "73"}):
            return token
        return None

    if fields[0] == "CQ":
        # A directed CQ has an extra region token: CQ DX / CQ 123 / CQ POTA.
        sender = fields[1] if call(fields[1]) else (fields[2] if len(fields) > 2 else "")
        return call(sender)
    if fields[0] in {"DE", "QRZ"}:
        return call(fields[1])
    if call(fields[0]) or fields[0] == "<...>":
        return call(fields[1])
    return None


def align_psk_decodes(
    ours: list[Decode], remote: list[Decode], signals: list[PskSignal], dial: int,
) -> list[tuple[Decode | None, Decode | None, PskSignal | None]]:
    """Attach exact or recent sender context; omit all unrelated PSK reports."""
    rows = [(local, server, None) for local, server in match_decodes(ours, remote)]
    for signal in sorted(signals, key=lambda s: (s.freq, s.call)):
        candidates = []
        for index, (local, server, attached) in enumerate(rows):
            if attached is not None:
                continue
            for priority, decode in enumerate((local, server)):
                if decode is not None and transmitting_call(decode.text) == signal.call:
                    candidates.append((priority, abs(dial + decode.freq - signal.freq), index))
        if candidates:
            index = min(candidates)[2]
            local, server, _ = rows[index]
            rows[index] = local, server, signal
    return rows


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
    snr = entry.get("band_snr")
    traffic = f"  {rate:,.2f}/min" if isinstance(rate, (int, float)) else ""
    if not traffic and isinstance(snr, (int, float)):
        traffic = f"  SNR {snr:+.1f} dB"
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
    screen.nodelay(False)
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


def ubersdr_frame_time(payload: bytes, fallback: float) -> float:
    """Return UberSDR's nanosecond audio timestamp, rejecting implausible values."""
    if len(payload) >= 8:
        timestamp = struct.unpack_from("<Q", payload)[0] / 1_000_000_000
        if abs(timestamp - fallback) < 300:
            return timestamp
    return fallback


def resolve_receiver_url(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ft8-nn-live-compare/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.geturl()


def metrics_url(receiver_url: str) -> str:
    """Return the metrics endpoint while preserving an OpenWebRX subpath."""
    return urllib.parse.urljoin(receiver_url.rstrip("/") + "/", "metrics.json")


def directory_entries(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize an UberSDR API response or a legacy generated directory."""
    if isinstance(document.get("entries"), list):
        return document["entries"]
    entries = []
    for instance in document.get("instances", []):
        if not instance.get("is_online") or not instance.get("digital_decodes") or not instance.get("public_url"):
            continue
        tuning = instance.get("tuning_range") or {}
        minimum = tuning.get("min_frequency", 10_000)
        maximum = tuning.get("max_frequency", 30_000_000)
        reports = (instance.get("pskreporter_rank") or {}).get("reports") or {}
        conditions = instance.get("band_conditions") or {}
        for band in set(reports) | set(conditions):
            dial = FT8_FREQUENCIES.get(band)
            if dial is None or not minimum <= dial <= maximum:
                continue
            report = reports.get(band) or {}
            entries.append(
                {
                    "receiver": {
                        "url": instance["public_url"],
                        "label": instance.get("callsign") or instance.get("name") or instance["public_url"],
                        "site": instance.get("location", ""),
                        "lat": instance.get("latitude"),
                        "lon": instance.get("longitude"),
                        "type": "UberSDR",
                        "version": instance.get("version"),
                    },
                    "band": band,
                    "ft8_decodes_total": report.get("day"),
                    "band_snr": conditions.get(band),
                    "capabilities": {
                        "audio_stream": "ubersdr-opus-websocket",
                        "decodes": "ubersdr-dxcluster-websocket",
                        "ft8_decoder": "ubersdr-digital-decodes",
                    },
                }
            )
    return sorted(
        entries,
        key=lambda entry: (
            -(entry.get("band_snr") if isinstance(entry.get("band_snr"), (int, float)) else -999),
            entry["receiver"]["url"],
            entry["band"],
        ),
    )


def load_directory(source: str, timeout: float) -> dict[str, Any]:
    if urllib.parse.urlparse(source).scheme in ("http", "https"):
        request = urllib.request.Request(source, headers={"User-Agent": "ft8-nn-live-compare/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    return json.loads(Path(source).read_text())


class OpusDecoder:
    """Thin libopus wrapper for UberSDR's header-plus-raw-Opus frames."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        names = [ctypes.util.find_library("opus")]
        if sys.platform == "darwin":
            names.extend(["/opt/local/lib/libopus.0.dylib", "/opt/homebrew/lib/libopus.dylib"])
        self.lib = None
        for name in names:
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError("libopus is required for UberSDR audio")
        self.lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int,
        ]
        self.lib.opus_decode.restype = ctypes.c_int
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        error = ctypes.c_int()
        self.decoder = self.lib.opus_decoder_create(sample_rate, channels, ctypes.byref(error))
        if not self.decoder or error.value:
            raise RuntimeError(f"opus_decoder_create failed: {error.value}")
        self.channels = channels

    def decode(self, packet: bytes) -> array.array:
        encoded = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
        pcm = (ctypes.c_int16 * (5760 * self.channels))()
        frames = self.lib.opus_decode(self.decoder, encoded, len(packet), pcm, 5760, 0)
        if frames < 0:
            raise RuntimeError(f"opus_decode failed: {frames}")
        result = array.array("h")
        result.frombytes(bytes(pcm)[: frames * self.channels * 2])
        return result

    def close(self) -> None:
        if self.decoder:
            self.lib.opus_decoder_destroy(self.decoder)
            self.decoder = None


class UberSDRClient:
    def __init__(
        self,
        url: str,
        band: str,
        on_audio: Callable[[array.array, float], None],
        on_decode: Callable[[int, Decode], None],
        on_status: Callable[[str], None],
        timeout: float = 10.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.band = band
        self.dial = FT8_FREQUENCIES[band]
        self.on_audio = on_audio
        self.on_decode = on_decode
        self.on_status = on_status
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        self.audio_ws: WebSocket | None = None
        self.spots_ws: WebSocket | None = None
        self.decoder: OpusDecoder | None = None
        self.threads: list[threading.Thread] = []
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.first_slot: int | None = None

    def start(self) -> None:
        self.first_slot = (int(time.time()) // 15 + 1) * 15
        request = urllib.request.Request(
            self.url + "/connection",
            data=json.dumps({"user_session_id": self.session_id}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "ft8-nn-live-compare/1"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            permission = json.load(response)
        if not permission.get("allowed"):
            raise RuntimeError(permission.get("reason", "connection rejected"))
        parsed = urllib.parse.urlparse(self.url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        query = urllib.parse.urlencode(
            {
                "frequency": self.dial,
                "mode": "usb",
                "user_session_id": self.session_id,
                "format": "opus",
                "version": 3,
            }
        )
        self.audio_ws = WebSocket(f"{scheme}://{parsed.netloc}/ws?{query}", self.timeout)
        self.spots_ws = WebSocket(
            f"{scheme}://{parsed.netloc}/ws/dxcluster?user_session_id={urllib.parse.quote(self.session_id)}",
            self.timeout,
        )
        self.audio_ws.send_json(
            {"type": "tune", "frequency": self.dial, "mode": "usb", "bandwidthLow": 0, "bandwidthHigh": 3000}
        )
        self.spots_ws.send_json({"type": "subscribe_digital_spots"})
        self.threads = [
            threading.Thread(target=self._audio_loop, name="ubersdr-audio", daemon=True),
            threading.Thread(target=self._spots_loop, name="ubersdr-spots", daemon=True),
        ]
        for thread in self.threads:
            thread.start()
        self.on_status(
            f"streaming UberSDR at {self.dial / 1e6:.6f} MHz; "
            "collecting first complete local slot (up to 30 s)"
        )

    def _fail(self, error: Exception) -> None:
        if not self.stop_event.is_set():
            self.error = error
            self.on_status(f"UberSDR error: {error}")

    def _audio_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                kind, payload = self.audio_ws.recv()
                if kind != "binary" or len(payload) < 22:
                    continue
                sample_rate = struct.unpack_from("<I", payload, 8)[0]
                channels = payload[12]
                if self.decoder is None:
                    if sample_rate != 12_000 or channels != 1:
                        raise RuntimeError(f"unsupported UberSDR audio format: {sample_rate} Hz, {channels} channels")
                    self.decoder = OpusDecoder(sample_rate, channels)
                samples = self.decoder.decode(payload[21:])
                if samples:
                    now = time.time()
                    self.on_audio(samples, ubersdr_frame_time(payload, now))
        except Exception as error:
            self._fail(error)

    def _spots_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                kind, payload = self.spots_ws.recv()
                if kind != "text":
                    continue
                message = json.loads(payload)
                value = message.get("data")
                if message.get("type") != "digital_spot" or not isinstance(value, dict):
                    continue
                if value.get("mode") != "FT8" or value.get("band") != self.band:
                    continue
                slot = int(parse_timestamp(value["timestamp"]) // 15 * 15)
                # The subscription begins with buffered historical spots. They cannot
                # have matching local audio from a session that has only just started.
                if self.first_slot is not None and slot < self.first_slot:
                    continue
                self.on_decode(
                    slot,
                    Decode(
                        audio_frequency(value.get("frequency", 0), self.dial),
                        float(value.get("dt", 0)),
                        float(value.get("snr", 0)),
                        str(value.get("message", "")),
                        "remote",
                    ),
                )
        except Exception as error:
            self._fail(error)

    def close(self) -> None:
        self.stop_event.set()
        for ws in (self.audio_ws, self.spots_ws):
            if ws:
                ws.close()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
        if self.decoder:
            self.decoder.close()


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
        self.psk: dict[int, dict[str, PskSignal]] = {}
        self.psk_status = "starting"
        self.selected_slot: int | None = None
        self.status = "starting"
        self.latest_audio_slot: int | None = None
        self.decode_seconds: dict[int, float] = {}
        self.error: str | None = None

    def add_psk(self, value: Any, band: str) -> None:
        report = parse_psk_report(value, band)
        if report is None:
            return
        slot, signal = report
        with self.lock:
            signals = self.psk.setdefault(slot, {})
            previous = signals.get(signal.call)
            if previous is None:
                signals[signal.call] = signal
            else:
                previous.reports += 1
                if signal.grid:
                    previous.grid = signal.grid
            # The feed is batched/out of order. Keep recent transmit slots separately
            # from local audio so late reports neither advance nor evict local results.
            for old in sorted(self.psk)[:-SLOT_HISTORY]:
                del self.psk[old]

    def psk_context(self, slot: int | None) -> list[PskSignal]:
        """Snapshot the latest report per sender at/before the viewed slot.

        Caller holds self.lock. Expiry follows transmission time, never arrival
        time or local decode activity, so delayed reports cannot revive old data.
        """
        if slot is None:
            return []
        latest: dict[str, PskSignal] = {}
        for report_slot in sorted(self.psk):
            if 0 <= slot - report_slot <= PSK_CONTEXT_SECONDS:
                for signal in self.psk[report_slot].values():
                    latest[signal.call] = PskSignal(
                        signal.call, signal.grid, signal.freq, signal.reports, report_slot)
        return list(latest.values())

    def view_slot(self) -> int | None:
        if self.selected_slot is not None:
            return self.selected_slot
        decodes = set(self.ours) | set(self.remote)
        return max(decodes) if decodes else None

    def navigate_slot(self, direction: int) -> None:
        with self.lock:
            slots = sorted(set(self.ours) | set(self.remote))
            current = self.view_slot()
            if current is None:
                return
            candidates = [slot for slot in slots if (slot - current) * direction > 0]
            if candidates:
                self.selected_slot = min(candidates) if direction > 0 else max(candidates)

    def trim(self) -> None:
        slots = sorted(set(self.ours) | set(self.remote))
        for old in slots[:-SLOT_HISTORY]:
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
    title = f"{receiver.get('label', receiver['url'])} — {entry['band']} FT8 — [/] slots, l live, q back"
    screen.addnstr(0, 0, title, width - 1, curses.A_BOLD if curses is not None else 0)
    with state.lock:
        status = state.status
        slot = state.view_slot()
        live = state.selected_slot is None
        ours = list(state.ours.get(slot, [])) if slot is not None else []
        remote = list(state.remote.get(slot, [])) if slot is not None else []
        elapsed = state.decode_seconds.get(slot) if slot is not None else None
        audio_slot = state.latest_audio_slot
        psk_status = state.psk_status
        psk_slot = slot
        psk = state.psk_context(slot)
    screen.addnstr(1, 0, status, width - 1)
    detail_url = (
        "https://instances.ubersdr.org/"
        if receiver.get("type") == "UberSDR"
        else metrics_url(receiver["url"])
    )
    screen.addnstr(2, 0, f"Directory: {detail_url}", width - 1)
    rate = entry.get("ft8_decodes_per_minute")
    rate_text = f"{rate:,.2f}/min" if isinstance(rate, (int, float)) else "unknown"
    decode_count = entry.get("ft8_decodes_total", entry.get("ft8_decodes"))
    count_text = f"{decode_count:,}" if isinstance(decode_count, int) else "unknown"
    traffic_label = "today" if receiver.get("type") == "UberSDR" else "total since restart"
    screen.addnstr(
        3, 0,
        f"FT8 traffic at directory build: {rate_text}  {traffic_label}: {count_text}",
        width - 1,
    )
    if slot is None:
        detail = "Waiting for the next UTC slot and remote decodes…"
    else:
        stamp = datetime.fromtimestamp(slot, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        detail = f"{'LIVE' if live else 'HISTORY'} Slot {stamp}  local {len(ours)}  remote {len(remote)}"
        if elapsed is not None:
            detail += f"  local decode {elapsed:.1f}s"
        if audio_slot is not None and audio_slot > slot:
            detail += "  (local decode pending for newer audio)"
    screen.addnstr(4, 0, detail, width - 1)
    rows = align_psk_decodes(ours, remote, psk, FT8_FREQUENCIES[entry["band"]])
    matched_psk = [signal for _, _, signal in rows if signal is not None]
    exact_psk = [signal for signal in matched_psk if signal.slot == slot]
    recent_count = len(matched_psk) - len(exact_psk)
    psk_stamp = datetime.fromtimestamp(psk_slot, timezone.utc).strftime("%H:%M:%S UTC") if psk_slot is not None else "waiting"
    screen.addnstr(5, 0, f"PSK Reporter: {psk_status} | aligned TX slot {psk_stamp} | {len(matched_psk)} matched stations, {sum(s.reports for s in exact_psk)} reports in slot, {recent_count} prior (up to 5m)", width - 1)

    gap = 2
    col = max(1, (width - 1 - 2 * gap) // 3)
    right = col + gap
    third = 2 * (col + gap)
    underline = curses.A_UNDERLINE if curses is not None else 0
    headings = ("OUR DECODER  Hz  dB   DT  message", "REMOTE DECODER  Hz  dB   DT  message", "PSK REPORTER  MHz  reports/prior age  call/grid")
    for x, heading in zip((0, right, third), headings):
        if x < width - 1:
            screen.addnstr(6, x, heading, min(col, width - x - 1), underline)
    for index, (local, server, signal) in enumerate(rows[:max(0, height - 8)]):
        report_text = ""
        if signal is not None:
            count = (str(signal.reports) if signal.slot == slot
                     else f"prev {slot - signal.slot}s")
            report_text = f"{signal.freq / 1e6:10.6f} {count:>9} {signal.call} {signal.grid}"
        for x, value in zip((0, right, third), (format_decode(local, col), format_decode(server, col), report_text)):
            if x < width - 1:
                screen.addnstr(7 + index, x, value, min(col, width - x - 1))
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
            state.status = f"decoding local slot {datetime.fromtimestamp(slot, timezone.utc):%H:%M:%S} UTC…"
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
                    state.status = f"local decoder found {len(result)} messages; streaming next slot"
                    state.trim()
            except Exception as error:
                with state.lock:
                    state.error = str(error)
                    state.status = f"local decoder error: {error}"

    worker = threading.Thread(target=decode_loop, name="ft8nn", daemon=True)
    worker.start()
    client_class = UberSDRClient if entry["receiver"].get("type") == "UberSDR" else OpenWebRXClient
    client = client_class(entry["receiver"]["url"], entry["band"], assembler.push, add_remote, status, args.timeout)
    def psk_status(text: str) -> None:
        with state.lock:
            state.psk_status = text

    psk_client = PskReporterClient(entry["band"], lambda value: state.add_psk(value, entry["band"]), psk_status, args.timeout)
    try:
        psk_client.start()
        try:
            client.start()
        except Exception as error:
            status(f"connection failed: {error} — press q to quit")
        while True:
            draw_comparison(screen, entry, state)
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("["), curses.KEY_LEFT):
                state.navigate_slot(-1)
            elif key in (ord("]"), curses.KEY_RIGHT):
                state.navigate_slot(1)
            elif key in (ord("l"), ord("L")):
                with state.lock:
                    state.selected_slot = None
            time.sleep(0.1)
    finally:
        psk_client.close()
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


def receiver_ui(screen: Any, entries: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    """Run the picker until dismissed; q in a comparison returns to the picker."""
    opened_comparison = False
    while True:
        selected = pick_receiver(screen, entries)
        if selected is None:
            return opened_comparison
        opened_comparison = True
        run_comparison(screen, selected, args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", default=DEFAULT_DIRECTORY, help="UberSDR API URL or local directory JSON")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--library", type=Path, default=default_library())
    parser.add_argument("--threads", type=int, default=0, help="local decoder threads; 0 uses performance cores")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--audio-delay", type=float, default=0.0, help="seconds from UTC boundary until its audio arrives")
    args = parser.parse_args(argv)
    document = load_directory(args.directory, args.timeout)
    entries = [entry for entry in directory_entries(document) if entry.get("band") in FT8_FREQUENCIES]
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

    opened_comparison = curses.wrapper(lambda screen: receiver_ui(screen, entries, args))
    return 0 if opened_comparison else 1


if __name__ == "__main__":
    raise SystemExit(main())
