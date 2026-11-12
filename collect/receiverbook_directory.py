#!/usr/bin/env python3
"""Build a directory of public OpenWebRX receivers with FT8 decoding.

Receiverbook's map is the source of receiver identity and location.  Each
OpenWebRX receiver is then contacted at its public ``metrics.json`` endpoint.
For every advertised FT8 band, the script also opens the receiver WebSocket,
selects a profile covering the FT8 dial frequency, starts its demodulator, and
waits for actual audio samples.  An entry is emitted only after that complete
streaming probe succeeds.  Two metrics snapshots measure a short-window FT8
decode rate used to rank the resulting entries.

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.live_compare import FT8_FREQUENCIES, OpenWebRXClient


SOURCE_URL = "https://www.receiverbook.de/map"
DEFAULT_OUTPUT = ROOT / "datasets" / "receiverbook-directory.json"
USER_AGENT = "ft8-nn receiverbook directory builder/1"
BAND_RE = re.compile(r"^(?:\d+(?:\.\d+)?)(?:m|cm|mm)$", re.IGNORECASE)


def fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_receiverbook(html: str) -> list[dict[str, Any]]:
    """Extract and flatten the JSON assigned to ``var receivers`` on /map."""
    match = re.search(r"\bvar\s+receivers\s*=\s*", html)
    if not match:
        raise ValueError("receiver list not found in Receiverbook map page")
    sites, _ = json.JSONDecoder().raw_decode(html, match.end())

    result = []
    for site in sites:
        coordinates = site.get("location", {}).get("coordinates", [])
        if len(coordinates) != 2:
            continue
        lon, lat = coordinates
        for receiver in site.get("receivers", []):
            if receiver.get("type") != "OpenWebRX" or not receiver.get("url"):
                continue
            result.append(
                {
                    "url": receiver["url"],
                    "label": receiver.get("label", site.get("label", "")),
                    "site": site.get("label", ""),
                    "lat": lat,
                    "lon": lon,
                    "type": receiver["type"],
                    **({"version": receiver["version"]} if receiver.get("version") else {}),
                }
            )
    return result


def metrics_url(receiver_url: str) -> str:
    return urllib.parse.urljoin(receiver_url.rstrip("/") + "/", "metrics.json")


def ft8_bands(metrics: dict[str, Any]) -> list[str]:
    """Return bands whose OpenWebRX WSJT metrics include an FT8 decoder."""
    decodes = metrics.get("wsjt", {}).get("decodes", {})
    bands = []
    if not isinstance(decodes, dict):
        return bands
    for band, modes in decodes.items():
        if (
            isinstance(band, str)
            and BAND_RE.fullmatch(band)
            and isinstance(modes, dict)
            and any(str(mode).upper() == "FT8" for mode in modes)
        ):
            bands.append(band.lower())
    return sorted(bands, key=band_sort_key, reverse=True)


def ft8_decode_counts(metrics: dict[str, Any]) -> dict[str, int]:
    """Return the current cumulative FT8 decode count for each valid band."""
    decodes = metrics.get("wsjt", {}).get("decodes", {})
    if not isinstance(decodes, dict):
        return {}
    counts = {}
    for band, modes in decodes.items():
        if not isinstance(band, str) or not BAND_RE.fullmatch(band) or not isinstance(modes, dict):
            continue
        missing = object()
        ft8 = next((value for mode, value in modes.items() if str(mode).upper() == "FT8"), missing)
        if ft8 is missing:
            continue
        count = ft8.get("count") if isinstance(ft8, dict) else None
        if isinstance(count, (int, float)) and not isinstance(count, bool) and count >= 0:
            counts[band.lower()] = int(count)
        else:
            counts[band.lower()] = 0
    return counts


def band_sort_key(band: str) -> float:
    value = float(re.match(r"\d+(?:\.\d+)?", band).group())
    suffix = band[len(str(int(value))) :] if value.is_integer() else re.sub(r"^[\d.]+", "", band)
    return value * {"m": 1.0, "cm": 0.01, "mm": 0.001}[suffix]


def can_stream_audio(receiver_url: str, band: str, timeout: float) -> tuple[bool, str | None]:
    """Select an FT8 profile and require at least one non-empty audio frame."""
    if band not in FT8_FREQUENCIES:
        return False, f"no known FT8 dial frequency for {band}"

    audio_received = threading.Event()
    client = OpenWebRXClient(
        receiver_url,
        band,
        on_audio=lambda samples, _received_at: audio_received.set() if samples else None,
        on_decode=lambda _slot, _decode: None,
        on_status=lambda _status: None,
        timeout=timeout,
    )
    try:
        deadline = time.monotonic() + timeout
        client.start()
        if not audio_received.wait(max(0, deadline - time.monotonic())):
            if client.error:
                raise client.error
            raise TimeoutError("timed out waiting for audio samples")
        return True, None
    except Exception as error:
        return False, str(error)
    finally:
        client.close()


def read_metrics(receiver: dict[str, Any], timeout: float) -> tuple[dict[str, Any] | None, float, str | None]:
    """Fetch one metrics snapshot and record when it completed."""
    try:
        metrics = json.loads(fetch(metrics_url(receiver["url"]), timeout))
        if not isinstance(metrics, dict):
            raise ValueError("metrics response is not an object")
        return metrics, time.monotonic(), None
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as error:
        return None, time.monotonic(), f"{receiver['url']}: metrics: {error}"


def take_metrics_snapshot(
    receivers: list[dict[str, Any]], timeout: float, workers: int
) -> tuple[dict[str, tuple[dict[str, Any], float]], list[str]]:
    snapshots: dict[str, tuple[dict[str, Any], float]] = {}
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(read_metrics, receiver, timeout): receiver for receiver in receivers}
        for future in concurrent.futures.as_completed(futures):
            receiver = futures[future]
            metrics, captured_at, error = future.result()
            if metrics is not None:
                snapshots[receiver["url"]] = (metrics, captured_at)
            if error:
                errors.append(error)
    return snapshots, errors


def decodes_per_minute(before: int | None, after: int, elapsed: float) -> float:
    """Calculate a counter rate, treating a reset or missing baseline as zero traffic."""
    if before is None or after < before or elapsed <= 0:
        return 0.0
    return round((after - before) * 60.0 / elapsed, 2)


def probe(
    receiver: dict[str, Any],
    timeout: float,
    before_metrics: dict[str, Any] | None,
    after_metrics: dict[str, Any],
    sample_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    bands = ft8_bands(after_metrics)
    before_counts = ft8_decode_counts(before_metrics or {})
    after_counts = ft8_decode_counts(after_metrics)

    entries = []
    errors = []
    for band in bands:
        streamable, error = can_stream_audio(receiver["url"], band, timeout)
        if not streamable:
            errors.append(f"{receiver['url']} [{band}]: {error}")
            continue
        entries.append(
            {
                "receiver": receiver,
                "band": band,
                "ft8_decodes_total": after_counts.get(band, 0),
                "ft8_decodes_per_minute": decodes_per_minute(
                    before_counts.get(band), after_counts.get(band, 0), sample_seconds
                ),
                "traffic_sample_seconds": round(sample_seconds, 2),
                "capabilities": {
                    "audio_stream": "openwebrx-websocket",
                    "decodes": "openwebrx-websocket",
                    "ft8_decoder": "openwebrx-wsjt-metric",
                    "stream_probe": "audio-samples-received",
                },
            }
        )
    return entries, errors


def build(source: str, timeout: float, workers: int, traffic_window: float = 60.0) -> tuple[dict[str, Any], list[str]]:
    if urllib.parse.urlparse(source).scheme in ("http", "https"):
        html = fetch(source, timeout).decode("utf-8", "replace")
    else:
        html = Path(source).read_text(encoding="utf-8")
    receivers = parse_receiverbook(html)

    before, errors = take_metrics_snapshot(receivers, timeout, workers)
    time.sleep(traffic_window)
    after, after_errors = take_metrics_snapshot(receivers, timeout, workers)
    errors.extend(after_errors)

    entries: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for receiver in receivers:
            url = receiver["url"]
            if url not in after:
                continue
            after_metrics, after_at = after[url]
            before_metrics, before_at = before.get(url, (None, after_at - traffic_window))
            sample_seconds = max(0.001, after_at - before_at)
            futures.append(
                executor.submit(
                    probe, receiver, timeout, before_metrics, after_metrics, sample_seconds
                )
            )
        for future in concurrent.futures.as_completed(futures):
            found, probe_errors = future.result()
            entries.extend(found)
            errors.extend(probe_errors)

    entries.sort(
        key=lambda entry: (
            -entry["ft8_decodes_per_minute"],
            entry["receiver"]["url"],
            -band_sort_key(entry["band"]),
        )
    )
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "criteria": {
            "internet_accessible": "public OpenWebRX WebSocket session connected",
            "ft8_decoder_installed": "wsjt.decodes contains an FT8 metric for the band",
            "audio_and_decodes_streamable": (
                "selected a profile covering the FT8 dial frequency and received audio samples"
            ),
            "traffic_order": "descending FT8 counter delta per minute",
            "traffic_measurement": (
                "two metrics.json snapshots; counter resets or missing baselines produce a zero rate"
            ),
        },
        "traffic_window_seconds": traffic_window,
        "entries": entries,
    }
    return document, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_URL, help="Receiverbook map URL or saved HTML file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=8.0, help="per-request timeout in seconds")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--traffic-window", type=float, default=60.0,
        help="seconds between metrics snapshots used to calculate FT8 decodes per minute",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.workers <= 0 or args.traffic_window <= 0:
        parser.error("--timeout, --workers, and --traffic-window must be positive")

    document, errors = build(args.source, args.timeout, args.workers, args.traffic_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(document['entries'])} receiver-band entries to {args.output} "
        f"({len(errors)} receiver-band probes failed)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
