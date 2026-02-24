import config as app_config
from utils import cpu_limiter


class _FakeProcess:
    def __init__(self, available):
        self.available = list(available)
        self.set_calls = []

    def cpu_affinity(self, cpus=None):
        if cpus is None:
            return list(self.available)
        self.set_calls.append(list(cpus))


def test_restrict_to_cpus_disabled_by_zero(monkeypatch):
    fake = _FakeProcess([2, 4, 6])
    monkeypatch.setattr(cpu_limiter, "get_config", lambda: {"CPU_LIMIT": 0})
    monkeypatch.setattr(cpu_limiter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpu_limiter.psutil, "Process", lambda _pid: fake)

    cpu_limiter.restrict_to_cpus()

    assert fake.set_calls == []


def test_restrict_to_cpus_uses_available_affinity_subset(monkeypatch):
    fake = _FakeProcess([2, 4, 6])
    monkeypatch.setattr(cpu_limiter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpu_limiter.psutil, "Process", lambda _pid: fake)

    cpu_limiter.restrict_to_cpus(cpu_limit=2)

    assert fake.set_calls == [[2, 4]]


def test_restrict_to_cpus_clamps_to_available_count(monkeypatch):
    fake = _FakeProcess([1, 5])
    monkeypatch.setattr(cpu_limiter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpu_limiter.psutil, "Process", lambda _pid: fake)

    cpu_limiter.restrict_to_cpus(cpu_limit=8)

    assert fake.set_calls == [[1, 5]]


def test_load_config_allows_cpu_limit_zero(monkeypatch):
    monkeypatch.setenv("CPU_LIMIT", "0")
    monkeypatch.setattr(app_config, "load_settings_yaml", lambda _output_dir: {})

    cfg = app_config._load_config()

    assert cfg["CPU_LIMIT"] == 0


def test_load_config_clamps_negative_cpu_limit_to_zero(monkeypatch):
    monkeypatch.setenv("CPU_LIMIT", "-3")
    monkeypatch.setattr(app_config, "load_settings_yaml", lambda _output_dir: {})

    cfg = app_config._load_config()

    assert cfg["CPU_LIMIT"] == 0
