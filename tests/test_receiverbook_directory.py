import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "collect" / "receiverbook_directory.py"
SPEC = importlib.util.spec_from_file_location("receiverbook_directory", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_receiverbook_flattens_only_openwebrx():
    html = '''<script>var receivers = [{"label":"site","location":{"coordinates":[1.5,2.5]},
      "receivers":[{"label":"owrx","url":"https://example.test/rx/","type":"OpenWebRX","version":"1.2"},
      {"label":"kiwi","url":"https://example.test/kiwi/","type":"KiwiSDR"}]}];</script>'''
    assert MODULE.parse_receiverbook(html) == [
        {
            "url": "https://example.test/rx/",
            "label": "owrx",
            "site": "site",
            "lat": 2.5,
            "lon": 1.5,
            "type": "OpenWebRX",
            "version": "1.2",
        }
    ]


def test_default_source_is_ubersdr_directory():
    assert MODULE.SOURCE_URL == "https://instances.ubersdr.org/api/instances?conditions=true"


def test_build_accepts_ubersdr_api_json(tmp_path):
    source = tmp_path / "instances.json"
    source.write_text('''{"instances":[{
      "is_online":true,"digital_decodes":true,
      "public_url":"https://example.tunnel.ubersdr.org/","callsign":"K1ABC",
      "tuning_range":{"min_frequency":10000,"max_frequency":30000000},
      "band_conditions":{"40m":12.5},
      "pskreporter_rank":{"reports":{"40m":{"day":42}}}
    }]}''')
    document, errors = MODULE.build(str(source), 1.0, 1)
    assert errors == []
    assert document["entries"][0]["receiver"]["type"] == "UberSDR"
    assert document["entries"][0]["band"] == "40m"


def test_ft8_bands_require_band_and_ft8_metric():
    metrics = {
        "wsjt": {
            "decodes": {
                "40m": {"FT8": {"count": 0}, "WSPR": {"count": 3}},
                "20m": {"ft8": {"count": 12}},
                "not-a-band": {"FT8": {"count": 1}},
                "10m": {"WSPR": {"count": 1}},
            }
        }
    }
    assert MODULE.ft8_bands(metrics) == ["40m", "20m"]
    assert MODULE.ft8_decode_counts(metrics) == {"40m": 0, "20m": 12}


def test_metrics_url_preserves_receiver_subpath():
    assert MODULE.metrics_url("https://example.test/openwebrx/") == "https://example.test/openwebrx/metrics.json"


def test_can_stream_audio_requires_samples(monkeypatch):
    class FakeClient:
        def __init__(self, _url, _band, on_audio, on_decode, on_status, timeout):
            self.on_audio = on_audio
            self.error = None

        def start(self):
            self.on_audio([1, 2], 0.0)

        def close(self):
            pass

    monkeypatch.setattr(MODULE, "OpenWebRXClient", FakeClient)
    assert MODULE.can_stream_audio("https://example.test/", "40m", 0.01) == (True, None)


def test_can_stream_audio_rejects_silent_stream(monkeypatch):
    class FakeClient:
        def __init__(self, _url, _band, on_audio, on_decode, on_status, timeout):
            self.error = None

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(MODULE, "OpenWebRXClient", FakeClient)
    available, error = MODULE.can_stream_audio("https://example.test/", "40m", 0.001)
    assert not available
    assert "audio samples" in error


def test_probe_keeps_only_streamable_bands(monkeypatch):
    before = {"wsjt": {"decodes": {"40m": {"FT8": {"count": 30}}, "20m": {"FT8": {"count": 0}}}}}
    after = {"wsjt": {"decodes": {"40m": {"FT8": {"count": 42}}, "20m": {"FT8": {"count": 1}}}}}
    monkeypatch.setattr(
        MODULE,
        "can_stream_audio",
        lambda _url, band, _timeout: (band == "40m", None if band == "40m" else "silent"),
    )
    receiver = {"url": "https://example.test/", "label": "test", "type": "OpenWebRX"}
    entries, errors = MODULE.probe(receiver, 1.0, before, after, 60.0)
    assert [entry["band"] for entry in entries] == ["40m"]
    assert entries[0]["ft8_decodes_total"] == 42
    assert entries[0]["ft8_decodes_per_minute"] == 12.0
    assert entries[0]["traffic_sample_seconds"] == 60.0
    assert entries[0]["capabilities"]["stream_probe"] == "audio-samples-received"
    assert errors == ["https://example.test/ [20m]: silent"]


def test_decodes_per_minute_handles_elapsed_time_and_counter_resets():
    assert MODULE.decodes_per_minute(100, 112, 60) == 12.0
    assert MODULE.decodes_per_minute(100, 112, 30) == 24.0
    assert MODULE.decodes_per_minute(100, 2, 60) == 0.0
    assert MODULE.decodes_per_minute(None, 112, 60) == 0.0


def test_build_sorts_by_ft8_traffic(monkeypatch, tmp_path):
    source = tmp_path / "map.html"
    source.write_text(
        '''<script>var receivers = [{"label":"site","location":{"coordinates":[1,2]},
        "receivers":[{"label":"rx","url":"https://example.test/","type":"OpenWebRX"}]}];</script>'''
    )
    entries = [
        {"receiver": {"url": "b"}, "band": "20m", "ft8_decodes_per_minute": 7},
        {"receiver": {"url": "a"}, "band": "40m", "ft8_decodes_per_minute": 100},
    ]
    snapshots = iter([
        ({"https://example.test/": ({}, 10.0)}, []),
        ({"https://example.test/": ({}, 70.0)}, []),
    ])
    monkeypatch.setattr(MODULE, "take_metrics_snapshot", lambda *_args: next(snapshots))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(MODULE, "probe", lambda *_args: (entries, []))
    document, errors = MODULE.build(str(source), 1.0, 1, 60.0)
    assert errors == []
    assert [entry["ft8_decodes_per_minute"] for entry in document["entries"]] == [100, 7]
    assert document["traffic_window_seconds"] == 60.0
