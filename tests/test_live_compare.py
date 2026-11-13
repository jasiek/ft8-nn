import array
import builtins
import importlib.util
import sys
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools" / "live_compare.py"
SPEC = importlib.util.spec_from_file_location("live_compare", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_module_imports_without_curses(monkeypatch):
    real_import = builtins.__import__

    def import_without_curses(name, *args, **kwargs):
        if name == "curses":
            raise ModuleNotFoundError("No module named '_curses'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_curses)
    spec = importlib.util.spec_from_file_location("live_compare_without_curses", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.curses is None
        assert module.normalize_message(" cq  k1abc ") == "CQ K1ABC"
    finally:
        del sys.modules[spec.name]


def decode(freq, text):
    return MODULE.Decode(freq, 0.1, -10, text)


def test_filter_entries_matches_multiple_substrings():
    entries = [
        {"band": "40m", "receiver": {"label": "Example Poland", "site": "Warsaw", "url": "https://one/"}},
        {"band": "20m", "receiver": {"label": "Example Spain", "site": "Madrid", "url": "https://two/"}},
    ]
    assert MODULE.filter_entries(entries, "pol 40") == entries[:1]
    assert MODULE.filter_entries(entries, "example") == entries


def test_match_decodes_pairs_messages_and_orders_by_frequency():
    ours = [decode(1400, "CQ K1ABC FN31"), decode(700, "W1AAA K2BBB -10")]
    remote = [decode(702, "W1AAA K2BBB -10"), decode(1100, "CQ ONLY DX10")]
    rows = MODULE.match_decodes(ours, remote)
    assert rows == [
        (ours[1], remote[0]),
        (None, remote[1]),
        (ours[0], None),
    ]


def test_match_decodes_normalizes_hashed_calls():
    ours = [decode(1000, "CQ <K1ABC> FN31")]
    remote = [decode(1002, "CQ <...> FN31")]
    assert MODULE.match_decodes(ours, remote) == [(ours[0], remote[0])]


def test_slot_assembler_aligns_and_emits_exact_slots():
    emitted = []
    assembler = MODULE.SlotAssembler(lambda slot, samples: emitted.append((slot, samples)), rate=4)
    assembler.push(array.array("h", range(4)), 14.0)
    assert emitted == []
    assembler.push(array.array("h", range(60)), 30.0)
    assert len(emitted) == 1
    assert emitted[0][0] == 15
    assert len(emitted[0][1]) == 60


def test_adpcm_sync_and_zero_nibbles():
    decoder = MODULE.ImaAdpcmDecoder()
    output = decoder.decode(b"SYNC" + b"\x00\x00\x00\x00" + b"\x00\x00")
    assert list(output) == [0, 0, 0, 0]


def test_parse_timestamp_iso_and_milliseconds():
    assert MODULE.parse_timestamp("1970-01-01T00:00:15Z") == 15
    assert MODULE.parse_timestamp(15_000) == 15_000
    assert MODULE.parse_timestamp(15_000_000_000) == 15_000_000


def test_remote_absolute_frequency_is_normalized_to_audio_offset():
    assert MODULE.audio_frequency(7_075_997, 7_074_000) == 1997
    assert MODULE.audio_frequency(1997, 7_074_000) == 1997


def test_ubersdr_frame_time_reads_nanoseconds_and_rejects_bad_values():
    payload = (1_700_000_000_250_000_000).to_bytes(8, "little")
    assert MODULE.ubersdr_frame_time(payload, 1_700_000_000.5) == 1_700_000_000.25
    assert MODULE.ubersdr_frame_time(payload, 1_800_000_000.0) == 1_800_000_000.0


def test_receiver_ui_returns_to_picker_after_comparison(monkeypatch):
    first = {"receiver": {"url": "https://one/"}}
    second = {"receiver": {"url": "https://two/"}}
    choices = iter([first, second, None])
    opened = []
    monkeypatch.setattr(MODULE, "pick_receiver", lambda _screen, _entries: next(choices))
    monkeypatch.setattr(MODULE, "run_comparison", lambda _screen, entry, _args: opened.append(entry))
    assert MODULE.receiver_ui(None, [first, second], object())
    assert opened == [first, second]


def test_metrics_url_preserves_receiver_subpath():
    assert MODULE.metrics_url("https://example.test/openwebrx/") == "https://example.test/openwebrx/metrics.json"


def test_ubersdr_directory_response_is_normalized():
    document = {
        "instances": [{
            "is_online": True,
            "digital_decodes": True,
            "public_url": "https://example.tunnel.ubersdr.org/",
            "callsign": "K1ABC",
            "location": "Example",
            "latitude": 1,
            "longitude": 2,
            "version": "0.1.68",
            "tuning_range": {"min_frequency": 10_000, "max_frequency": 30_000_000},
            "band_conditions": {"40m": 22.5, "6m": 30},
            "pskreporter_rank": {"reports": {"40m": {"day": 1234}}},
        }]
    }
    entries = MODULE.directory_entries(document)
    assert len(entries) == 1
    assert entries[0]["band"] == "40m"
    assert entries[0]["ft8_decodes_total"] == 1234
    assert entries[0]["receiver"]["type"] == "UberSDR"


def test_streaming_view_shows_metrics_link_and_decode_snapshot():
    class Screen:
        def __init__(self):
            self.lines = {}

        def erase(self):
            pass

        def getmaxyx(self):
            return 24, 160

        def addnstr(self, row, _column, value, _length, *_attrs):
            self.lines[row] = value

        def refresh(self):
            pass

    screen = Screen()
    entry = {
        "band": "40m",
        "ft8_decodes_total": 1_234_567,
        "ft8_decodes_per_minute": 42.5,
        "receiver": {"url": "https://example.test/rx/", "label": "Example"},
    }
    MODULE.draw_comparison(screen, entry, MODULE.ComparisonState())
    assert screen.lines[2] == "Directory: https://example.test/rx/metrics.json"
    assert screen.lines[3] == "FT8 traffic at directory build: 42.50/min  total since restart: 1,234,567"
    assert "42.50/min" in MODULE.entry_label(entry)
