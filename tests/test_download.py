"""Tests for the download script's certificate handling.

The failure these cover only happens on a Windows machine that has not yet cached
GroupLens's root certificate, so it cannot be reproduced on a machine that has. The system
store's rejection is simulated instead.
"""

import ssl
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_data


class FakeResponse:
    def __init__(self, context):
        self.context = context


def verification_failure():
    return urllib.error.URLError(
        ssl.SSLCertVerificationError(1, "unable to get local issuer certificate")
    )


def test_a_trusted_chain_uses_the_system_store(monkeypatch):
    calls = []

    def fake_urlopen(url, context=None):
        calls.append(context)
        return FakeResponse(context)

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fake_urlopen)
    response = download_data.open_url("https://example.invalid/file")
    assert calls == [None]
    assert response.context is None


def test_an_uncached_root_falls_back_to_certifi(monkeypatch):
    """The fresh-machine case: the system store rejects, the retry must succeed."""
    calls = []

    def fake_urlopen(url, context=None):
        calls.append(context)
        if context is None:
            raise verification_failure()
        return FakeResponse(context)

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fake_urlopen)
    response = download_data.open_url("https://example.invalid/file")
    assert len(calls) == 2
    assert calls[0] is None
    assert isinstance(response.context, ssl.SSLContext)


def test_the_fallback_still_verifies_certificates(monkeypatch):
    """The retry swaps the trust list; it must never switch verification off."""
    captured = []

    def fake_urlopen(url, context=None):
        if context is None:
            raise verification_failure()
        captured.append(context)
        return FakeResponse(context)

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fake_urlopen)
    download_data.open_url("https://example.invalid/file")
    assert captured[0].verify_mode == ssl.CERT_REQUIRED
    assert captured[0].check_hostname is True


def test_a_non_certificate_failure_is_not_retried(monkeypatch):
    """A DNS failure or refused connection has nothing to do with trust lists."""
    calls = []

    def fake_urlopen(url, context=None):
        calls.append(context)
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        download_data.open_url("https://example.invalid/file")
    assert len(calls) == 1


def test_a_chain_neither_store_trusts_still_fails(monkeypatch):
    def fake_urlopen(url, context=None):
        raise verification_failure()

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        download_data.open_url("https://example.invalid/file")


def test_the_pinned_certificate_workaround_is_gone():
    """GroupLens renewed; the expired-certificate path must not linger."""
    source = Path(download_data.__file__).read_text(encoding="utf-8")
    assert "allow_expired" not in source
    assert "CERT_NONE" not in source
    assert "check_hostname = False" not in source
