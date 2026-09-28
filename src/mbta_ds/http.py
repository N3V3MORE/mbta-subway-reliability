"""Shared HTTP helpers: one retrying session, safe downloads, disk caching.

Every network call in the project goes through here so that retry policy,
rate-limit handling and caching behave identically everywhere.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

USER_AGENT = "mbta-ds-poc/0.1 (CS 506 course project)"


def make_session(api_key: str | None = None) -> requests.Session:
    """Build a ``requests.Session`` with retries and rate-limit backoff.

    Retries cover connection errors and the transient status codes MBTA and
    ArcGIS emit under load (429 / 5xx).
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    if api_key:
        session.headers["X-API-Key"] = api_key

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(
    url: str,
    session: requests.Session,
    *,
    params: dict | None = None,
    timeout: int = 60,
    attempts: int = 3,
) -> dict:
    """GET a URL and decode JSON, with a bounded retry loop on timeouts.

    ``requests``' built-in retries do not cover read timeouts on some ArcGIS
    endpoints, which are slow enough to need their own retry.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_error = exc
            if attempt < attempts:
                sleep_for = 2.0 * attempt
                log.warning("GET %s failed (%s); retrying in %.0fs", url, exc, sleep_for)
                time.sleep(sleep_for)
    raise RuntimeError(f"GET {url} failed after {attempts} attempts") from last_error


def download(
    url: str,
    dest: Path,
    session: requests.Session,
    *,
    skip_if_exists: bool = True,
    timeout: int = 180,
) -> Path:
    """Download ``url`` to ``dest`` atomically, skipping files already cached.

    Writes to a ``.part`` sibling first so an interrupted run never leaves a
    truncated file that a later run would treat as a valid cache hit.
    """
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        log.debug("cache hit %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # `with` releases the pooled connection even on error; a streamed response
    # that is never closed holds its connection until garbage collection.
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        try:
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
            tmp.replace(dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return dest
