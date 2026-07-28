"""Byte-range sources for reading an MCAP without downloading it.

All MCAP inspection in this package goes through :class:`ByteSource`, which exposes a file size and
ranged reads. Local files, HTTPS URLs (including presigned S3 URLs), and ``s3://`` paths all satisfy
it, so registration code never needs to know where the bytes live.
"""

from __future__ import annotations

import logging
import re
from types import TracebackType
from typing import Any, BinaryIO, Protocol, Tuple, Type, runtime_checkable

import requests

logger = logging.getLogger(__name__)

_S3_URI = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0


class McapSourceError(RuntimeError):
    """Raised when the bytes of an MCAP cannot be reached or have changed underneath us."""


@runtime_checkable
class ByteSource(Protocol):
    """A random-access, read-only view over a remote or local object."""

    @property
    def uri(self) -> str:
        """Where the bytes live, for logging and error messages."""
        ...

    @property
    def size(self) -> int:
        """Total size of the object in bytes."""
        ...

    @property
    def etag(self) -> str | None:
        """Entity tag of the object, when the backing store exposes one.

        Registration pins this so that a later read fails loudly if the customer overwrites the
        object rather than silently serving frames from a different recording.
        """
        ...

    def read_range(self, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``offset``."""
        ...

    def close(self) -> None: ...


class _ByteSourceBase:
    """Shared context-manager plumbing and range validation."""

    _uri: str
    _size: int
    _etag: str | None = None
    bytes_read: int = 0
    range_requests: int = 0

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def size(self) -> int:
        return self._size

    @property
    def etag(self) -> str | None:
        return self._etag

    def _clamp(self, offset: int, length: int) -> Tuple[int, int]:
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        offset = min(offset, self._size)
        length = min(length, self._size - offset)
        return offset, length

    def _account(self, data: bytes) -> bytes:
        self.range_requests += 1
        self.bytes_read += len(data)
        return data

    def close(self) -> None:  # pragma: no cover - overridden where needed
        pass

    def __enter__(self) -> "_ByteSourceBase":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class FileByteSource(_ByteSourceBase):
    """Ranged reads over a local file or any seekable binary stream."""

    def __init__(self, stream: BinaryIO, uri: str = "<stream>", *, close_stream: bool = False) -> None:
        if not stream.seekable():
            raise ValueError("MCAP inspection requires a seekable stream")
        self._stream = stream
        self._uri = uri
        self._close_stream = close_stream
        current = stream.tell()
        self._size = stream.seek(0, 2)
        stream.seek(current)

    @classmethod
    def from_path(cls, path: str) -> "FileByteSource":
        return cls(open(path, "rb"), uri=path, close_stream=True)

    def read_range(self, offset: int, length: int) -> bytes:
        offset, length = self._clamp(offset, length)
        if length == 0:
            return b""
        self._stream.seek(offset)
        return self._account(self._stream.read(length))

    def close(self) -> None:
        if self._close_stream:
            self._stream.close()


class HttpRangeByteSource(_ByteSourceBase):
    """Ranged reads over an HTTP(S) URL, including presigned S3 URLs.

    The object's length and entity tag come from the response to the first ranged read rather than a
    separate ``HEAD``, because presigned URLs are commonly signed for ``GET`` only.
    """

    def __init__(
        self,
        url: str,
        *,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        expected_etag: str | None = None,
    ) -> None:
        self._uri = url
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._timeout = timeout
        self._expected_etag = expected_etag
        self._size = -1
        self._probe()

    def _probe(self) -> None:
        # A one-byte ranged GET returns Content-Range: bytes 0-0/<total>, which gives us the size
        # without needing HEAD permission on the object.
        response = self._request(0, 1)
        content_range = response.headers.get("Content-Range")
        if not content_range or "/" not in content_range:
            raise McapSourceError(
                f"{self._uri} does not support HTTP range requests "
                "(no Content-Range header); direct MCAP playback requires a range-capable store"
            )
        total = content_range.rsplit("/", 1)[-1].strip()
        if total == "*":
            raise McapSourceError(f"{self._uri} did not report a total object size")
        self._size = int(total)
        self._etag = _normalize_etag(response.headers.get("ETag"))
        self._verify_etag()

    def _verify_etag(self) -> None:
        if self._expected_etag is None or self._etag is None:
            return
        if self._etag != self._expected_etag:
            raise McapSourceError(
                f"{self._uri} changed since it was registered "
                f"(expected ETag {self._expected_etag}, found {self._etag}). Re-register the file."
            )

    def _request(self, offset: int, length: int) -> requests.Response:
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        try:
            response = self._session.get(self._uri, headers=headers, timeout=self._timeout, stream=False)
        except requests.RequestException as exc:
            raise McapSourceError(f"failed to read bytes {offset}..{offset + length} from {self._uri}: {exc}") from exc
        if response.status_code not in (200, 206):
            raise McapSourceError(
                f"unexpected status {response.status_code} reading bytes {offset}..{offset + length} from {self._uri}"
            )
        return response

    def read_range(self, offset: int, length: int) -> bytes:
        if self._size >= 0:
            offset, length = self._clamp(offset, length)
        if length == 0:
            return b""
        return self._account(self._request(offset, length).content)

    def close(self) -> None:
        if self._owns_session:
            self._session.close()


class S3ByteSource(_ByteSourceBase):
    """Ranged reads over ``s3://bucket/key`` using the caller's own boto3 credentials.

    ``boto3`` is not a dependency of this SDK. Install it (``pip install boto3``) to register MCAPs
    straight out of a bucket, or pass a presigned HTTPS URL instead.
    """

    def __init__(self, uri: str, *, client: Any = None, expected_etag: str | None = None) -> None:
        match = _S3_URI.match(uri)
        if match is None:
            raise ValueError(f"not an s3 uri: {uri!r}")
        self._uri = uri
        self._bucket = match.group("bucket")
        self._key = match.group("key")
        self._client = client if client is not None else _default_s3_client()
        head = self._client.head_object(Bucket=self._bucket, Key=self._key)
        self._size = int(head["ContentLength"])
        self._etag = _normalize_etag(head.get("ETag"))
        if expected_etag is not None and self._etag != expected_etag:
            raise McapSourceError(
                f"{uri} changed since it was registered (expected ETag {expected_etag}, found {self._etag}). "
                "Re-register the file."
            )

    def read_range(self, offset: int, length: int) -> bytes:
        offset, length = self._clamp(offset, length)
        if length == 0:
            return b""
        response = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={offset}-{offset + length - 1}"
        )
        return self._account(bytes(response["Body"].read()))


def _default_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise McapSourceError(
            "reading s3:// paths directly requires boto3 (pip install boto3), or pass a presigned https:// URL instead"
        ) from exc
    return boto3.client("s3")


def _normalize_etag(raw: str | None) -> str | None:
    if raw is None:
        return None
    return raw.strip().strip('"')


def open_byte_source(uri: str, *, expected_etag: str | None = None) -> ByteSource:
    """Open the right :class:`ByteSource` for ``uri``.

    Supports local paths, ``file://``, ``http(s)://`` (including presigned S3 URLs) and ``s3://``.
    """
    if uri.startswith(("http://", "https://")):
        return HttpRangeByteSource(uri, expected_etag=expected_etag)
    if uri.startswith("s3://"):
        return S3ByteSource(uri, expected_etag=expected_etag)
    if uri.startswith("file://"):
        return FileByteSource.from_path(uri[len("file://") :])
    return FileByteSource.from_path(uri)
