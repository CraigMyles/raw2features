"""Read-only HTTP Zarr stores whose query applies to every child request."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from fsspec.implementations.http import HTTPFileSystem
from zarr.storage import FsspecStore


class _QueryHTTPFileSystem(HTTPFileSystem):
    """Attach one raw query string to keyed reads below a query-free store root."""

    def __init__(self, query: str, **kwargs) -> None:
        self._raw_query = query
        # Signed queries are sensitive to percent-encoding. ``encoded=True`` tells
        # aiohttp/fsspec that the supplied URL is already encoded and must not turn
        # (for example) ``%2F`` into ``/`` or ``%252F``.
        kwargs.setdefault("asynchronous", True)
        kwargs.setdefault("encoded", True)
        kwargs.setdefault("skip_instance_cache", True)
        super().__init__(**kwargs)

    def _with_query(self, url: str) -> str:
        parsed = urlsplit(url)
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                self._raw_query,
                parsed.fragment,
            )
        )

    def encode_url(self, url):
        # This is HTTPFileSystem's single request boundary: keyed reads, ranges,
        # info/exists, listings, and file opens all pass through it. Keeping the
        # override here avoids depending on fsspec method signatures that changed
        # across the supported release range.
        return super().encode_url(self._with_query(str(url)))

    def close(self) -> None:
        if self._session is not None:
            self.close_session(self.loop, self._session)
            self._session = None


def query_http_store(uri: str) -> FsspecStore:
    """Build a read-only store that applies *uri*'s raw query to every key.

    Zarr joins keys internally. Giving it ``https://host/a.zarr?token=x`` directly
    therefore produces ``...a.zarr?token=x/zarr.json``. This store keeps a query-free
    root for key joining, then attaches the original query to each HTTP request. URI
    fragments are deliberately omitted from requests, as required by HTTP.
    """

    value = str(uri)
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError(f"query_http_store requires HTTP(S), got {parsed.scheme!r}")
    raw_scheme = value.partition(":")[0]
    root = urlunsplit((raw_scheme, parsed.netloc, parsed.path, "", ""))
    fs = _QueryHTTPFileSystem(parsed.query)
    return FsspecStore(fs=fs, path=root, read_only=True)
