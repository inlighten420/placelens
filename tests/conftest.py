import pytest


@pytest.fixture(autouse=True)
def _offline_map(monkeypatch):
    """Tests read their own OpenStreetMap test data; none reach out to Overture over the network."""
    monkeypatch.setenv("PLACELENS_FOOTPRINTS", "osm")


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """Tests run the same on any machine (a judge's, CI) and never spend real credits: dummy keys, never .env's."""
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
