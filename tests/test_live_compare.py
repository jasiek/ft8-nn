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


def spot(call="K1ABC", slot=150, **overrides):
    return dict({"md": "FT8", "b": "40m", "t_tx": slot, "t": slot + 12,
                 "sc": call, "sl": "FN31", "f": 7_075_000}, **overrides)


def test_psk_counts_reports_per_station_and_transmit_slot():
    state = MODULE.ComparisonState()
    state.add_psk(spot(), "40m")
    state.add_psk(spot(rc="W1AAA", t=169, f=7_075_003), "40m")
    state.add_psk(spot(slot=165), "40m")
    state.add_psk(spot(call="W2BBB"), "40m")
    assert state.psk[150]["K1ABC"].reports == 2
    assert state.psk[165]["K1ABC"].reports == 1
    assert state.psk[150]["W2BBB"].reports == 1
    # A delayed report updates its original slot, not the latest slot.
    state.add_psk(spot(t=999), "40m")
    assert state.psk[150]["K1ABC"].reports == 3
    assert not state.ours and not state.remote


def test_psk_filters_bad_reports_and_falls_back_to_receiver_time():
    for value in (None, [], {}, spot(md="FT4"), spot(b="20m"),
                  spot(t_tx="bad"), spot(f=float("nan")), spot(t_tx=float("inf")),
                  spot(sc=""), spot(sc=None)):
        assert MODULE.parse_psk_report(value, "40m") is None
    assert MODULE.parse_psk_report(spot(t_tx=None, t=179), "40m")[0] == 165
    assert MODULE.parse_psk_report(spot(t_tx=0), "40m")[0] == 0


def test_psk_retains_history_and_late_old_reports_do_not_evict_new_slots():
    state = MODULE.ComparisonState()
    for slot in range(0, (MODULE.SLOT_HISTORY + 2) * 15, 15):
        state.add_psk(spot(slot=slot), "40m")
    state.add_psk(spot(slot=0), "40m")
    assert sorted(state.psk) == list(range(30, (MODULE.SLOT_HISTORY + 2) * 15, 15))


def test_psk_column_hides_unmatched_reports_and_handles_narrow_terminal():
    class Screen:
        width = 180
        def __init__(self):
            self.cells = {}
        def erase(self):
            self.cells.clear()
        def getmaxyx(self):
            return 24, self.width
        def addnstr(self, row, column, value, length, *_attrs):
            assert column >= 0 and 0 < length <= self.width - column - 1
            self.cells[row, column] = value[:length]
        def refresh(self):
            pass
    screen = Screen()
    entry = {"band": "40m", "receiver": {"url": "https://example.test/"}}
    state = MODULE.ComparisonState()
    state.add_psk(spot(), "40m")
    state.add_psk(spot(), "40m")
    MODULE.draw_comparison(screen, entry, state)
    assert any("PSK REPORTER" in value for value in screen.cells.values())
    assert not any("K1ABC FN31" in value for value in screen.cells.values())
    assert "0 matched stations, 0 reports" in screen.cells[5, 0]
    state.ours[150] = [decode(1000, "CQ K1ABC FN31")]
    MODULE.draw_comparison(screen, entry, state)
    assert any("K1ABC FN31" in value and "2" in value for value in screen.cells.values())
    assert "00:02:30 UTC" in screen.cells[5, 0]
    for width in (80, 40, 10):
        screen.width = width
        MODULE.draw_comparison(screen, entry, state)


def test_mqtt_subscribes_and_reads_fragmented_publish(monkeypatch):
    import json
    import threading
    client = None
    received = threading.Event()
    payload = json.dumps(spot()).encode()
    topic = b"pskr/filter/v2/40m/FT8/K1ABC/W1AAA/FN31/FN32/291/291"
    publish = MODULE.PskReporterClient._packet(0x30, len(topic).to_bytes(2, "big") + topic + payload)

    class Socket:
        def __init__(self):
            self.sent = []
            self.chunks = iter([b"\x20", b"\x02\x00\x00", b"\x90\x03\x00\x01\x00" + publish[:2], publish[2:17], publish[17:]])
        def settimeout(self, timeout):
            pass
        def sendall(self, packet):
            self.sent.append(packet)
        def recv(self, size):
            try:
                return next(self.chunks)
            except StopIteration:
                client.stop.wait(0.01)
                raise MODULE.socket.timeout()
        def shutdown(self, how):
            pass
        def close(self):
            pass
    sock = Socket()
    class Context:
        def wrap_socket(self, raw, server_hostname):
            return raw
    monkeypatch.setattr(MODULE.socket, "create_connection", lambda *args: sock)
    monkeypatch.setattr(MODULE.ssl, "create_default_context", Context)
    reports, statuses = [], []
    def on_report(value):
        reports.append(value)
        received.set()
    client = MODULE.PskReporterClient("40m", on_report, statuses.append)
    client.start()
    try:
        assert received.wait(2)
    finally:
        client.close()
    assert reports == [spot()]
    assert "connected" in statuses
    assert b"pskr/filter/v2/40m/FT8/#" in sock.sent[1]
    assert not client.thread.is_alive()


def test_transmitting_call_uses_sender_not_recipient():
    cases = {
        "CQ K1ABC FN31": "K1ABC", "CQ DX K1ABC FN31": "K1ABC",
        "CQ 123 K1ABC FN31": "K1ABC", "W2BBB K1ABC -10": "K1ABC",
        "DE K1ABC FN31": "K1ABC", "QRZ K1ABC FN31": "K1ABC",
        "<W2BBB> <K1ABC/P> R-12": "K1ABC/P",
        "<...> K1ABC RR73": "K1ABC", "K1ABC <...> -10": None,
        "CQ <...> FN31": None, "TNX 73 GL": None, "": None,
        "K1ABC RR73; W2BBB <KH1/KH7Z> -08": "KH1/KH7Z",
        "TU; W2BBB K1ABC R 579 MA": "K1ABC",
    }
    for text, expected in cases.items():
        assert MODULE.transmitting_call(text) == expected, text


def test_psk_aligns_by_sender_not_proximity_and_hides_unmatched_reports():
    local = [decode(1000, "CQ K1ABC FN31"), decode(1100, "K1ABC W2BBB -10")]
    remote = [decode(1500, "CQ W3CCC FN32")]
    a = MODULE.PskSignal("K1ABC", "FN31", 7_075_098, 3)
    b = MODULE.PskSignal("W2BBB", "FN31", 7_075_002, 2)
    c = MODULE.PskSignal("W3CCC", "FN32", 7_075_510, 4)
    other = MODULE.PskSignal("W4DDD", "FN33", 7_074_900, 1)
    rows = MODULE.align_psk_decodes(local, remote, [a, b, c, other], 7_074_000)
    assert rows == [(local[0], None, a),
                    (local[1], None, b), (None, remote[0], c)]


def test_psk_frequency_breaks_tie_but_does_not_match_unknown_sender():
    local = [decode(500, "CQ K1ABC FN31"), decode(1500, "CQ K1ABC FN31"),
             decode(2000, "CQ <...> FN31")]
    signal = MODULE.PskSignal("K1ABC", "FN31", 7_075_510, 3)
    unknown = MODULE.PskSignal("W2BBB", "FN31", 7_076_000, 2)
    rows = MODULE.align_psk_decodes(local, [], [signal, unknown], 7_074_000)
    assert (local[1], None, signal) in rows
    assert (local[0], None, None) in rows
    assert (local[2], None, None) in rows
    assert len(rows) == len(local)
    assert all(signal != unknown for _, _, signal in rows)


def test_slot_navigation_and_late_reports_keep_slots_separate():
    state = MODULE.ComparisonState()
    state.ours[150] = [decode(1000, "CQ K1ABC FN31")]
    state.ours[165] = [decode(1000, "CQ K1ABC FN31")]
    state.add_psk(spot(slot=150), "40m")
    state.add_psk(spot(slot=135), "40m")
    state.add_psk(spot(slot=195), "40m")
    assert state.view_slot() == 165
    assert not state.psk.get(state.view_slot())
    state.navigate_slot(-1)
    assert state.view_slot() == 150
    state.navigate_slot(-1)
    assert state.view_slot() == 150  # PSK-only slot 135 is hidden.
    state.add_psk(spot(slot=150), "40m")
    state.ours[180] = []
    assert state.view_slot() == 150
    assert state.psk[150]["K1ABC"].reports == 2
    state.navigate_slot(1)
    assert state.view_slot() == 165
    state.selected_slot = None
    assert state.view_slot() == 180


def test_rendered_psk_row_aligns_and_excludes_other_slots():
    class Screen:
        def __init__(self):
            self.cells = {}
        def erase(self):
            self.cells.clear()
        def getmaxyx(self):
            return 24, 240
        def addnstr(self, row, column, value, length, *_attrs):
            self.cells[row, column] = value[:length]
        def refresh(self):
            pass

    state = MODULE.ComparisonState()
    state.ours[150] = [decode(500, "CQ W2BBB FN32"), decode(1000, "CQ K1ABC FN31")]
    state.add_psk(spot(slot=150), "40m")
    state.add_psk(spot(call="W2BBB", slot=165), "40m")
    screen = Screen()
    entry = {"band": "40m", "receiver": {"url": "https://example.test/"}}
    MODULE.draw_comparison(screen, entry, state)
    # The newer PSK slot must not attach W2BBB's spot to the older local decode.
    assert not any(x > 0 and "W2BBB" in value for (y, x), value in screen.cells.items() if y >= 7)
    local_row = next(y for (y, x), value in screen.cells.items() if x == 0 and "CQ K1ABC" in value)
    assert any(y == local_row and x > 0 and "K1ABC" in value
               for (y, x), value in screen.cells.items())


def test_psk_context_survives_qso_and_expires_by_transmit_time():
    state = MODULE.ComparisonState()
    state.add_psk(spot(slot=150), "40m")
    state.add_psk(spot(slot=150), "40m")
    for text in ("W2BBB K1ABC -10", "W2BBB K1ABC R-12", "W2BBB K1ABC RR73", "W2BBB K1ABC 73"):
        local = decode(1000, text)
        rows = MODULE.align_psk_decodes([local], [], state.psk_context(180), 7_074_000)
        assert rows[0][2].call == "K1ABC"
        assert rows[0][2].slot == 150
        assert rows[0][2].reports == 2
    assert len(state.psk_context(150 + MODULE.PSK_CONTEXT_SECONDS)) == 1
    assert state.psk_context(150 + MODULE.PSK_CONTEXT_SECONDS + 15) == []
    # Arrival of another delayed report must not extend its lifetime.
    state.add_psk(spot(slot=150), "40m")
    assert state.psk_context(150 + MODULE.PSK_CONTEXT_SECONDS + 15) == []


def test_psk_context_prefers_exact_slot_and_never_looks_forward():
    state = MODULE.ComparisonState()
    state.add_psk(spot(slot=150), "40m")
    snapshot = state.psk_context(165)
    state.add_psk(spot(slot=150), "40m")
    assert snapshot[0].reports == 1  # Independent snapshot for UI drawing.
    state.add_psk(spot(slot=180, f=7_075_020), "40m")
    assert state.psk_context(180)[0].slot == 180
    assert state.psk_context(180)[0].reports == 1  # No count carried from the CQ.
    assert state.psk_context(165)[0].slot == 150
    assert state.psk_context(135) == []
    assert state.psk_context(None) == []


def test_psk_context_keeps_filtering_sender_and_remote_only_decodes():
    state = MODULE.ComparisonState()
    state.add_psk(spot(slot=150), "40m")
    state.add_psk(spot(slot=150, b="20m"), "40m")
    context = state.psk_context(180)
    remote = decode(1000, "W2BBB K1ABC -10")
    assert MODULE.align_psk_decodes([], [remote], context, 7_074_000)[0][2].call == "K1ABC"
    local = decode(1000, "K1ABC W2BBB -10")
    assert MODULE.align_psk_decodes([local], [], context, 7_074_000) == [(local, None, None)]


def test_rendered_qso_shows_prior_age_then_exact_slot_count():
    class Screen:
        def __init__(self):
            self.cells = {}
        def erase(self):
            self.cells.clear()
        def getmaxyx(self):
            return 24, 240
        def addnstr(self, row, column, value, length, *_attrs):
            self.cells[row, column] = value[:length]
        def refresh(self):
            pass

    screen = Screen()
    state = MODULE.ComparisonState()
    entry = {"band": "40m", "receiver": {"url": "https://example.test/"}}
    state.ours[180] = [decode(1000, "W2BBB K1ABC -10")]
    state.add_psk(spot(slot=150), "40m")
    MODULE.draw_comparison(screen, entry, state)
    assert any("prev 30s" in value and "K1ABC" in value for value in screen.cells.values())
    assert "0 reports in slot, 1 prior" in screen.cells[5, 0]
    state.add_psk(spot(slot=180), "40m")
    state.add_psk(spot(slot=180), "40m")
    MODULE.draw_comparison(screen, entry, state)
    assert not any("prev 30s" in value for value in screen.cells.values())
    assert "2 reports in slot, 0 prior" in screen.cells[5, 0]
