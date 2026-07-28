"""Wire types and client for registering direct-playback MCAP video with Nominal.

Registration tells Nominal three things about a customer-owned MCAP: where it lives, which of its
topics are video, and the file's own chunk index so playback can turn a timestamp into a byte range.
No video bytes are uploaded.

These types are hand-written rather than generated because the endpoint is newer than the published
``nominal-api`` wheel. The SDK follows the same escape hatch as ``ProtoWriteService`` in
``nominal.core._clientsbunch``: a ``conjure_python_client.Service`` subclass issuing the request
directly. When the endpoint reaches a published ``nominal-api`` release, this module can be replaced
by the generated client with no change to the public SDK surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from conjure_python_client import ConjureHTTPError, Service

from nominal.core._mcap.bitstream import BitstreamFormat, VideoCodec
from nominal.core._mcap.video import (
    ChunkRange,
    DirectPlaybackSupport,
    McapVideoChannel,
    MessageEncoding,
)
from nominal.core.exceptions import NominalError

logger = logging.getLogger(__name__)

DIRECT_MCAP_REGISTER_PATH = "/video/v1/direct-mcap/register"
DIRECT_MCAP_MANIFEST_PATH = "/video/v1/direct-mcap/playback-manifest"


class DirectMcapUnsupportedError(NominalError):
    """Raised when the Nominal deployment does not expose the direct-MCAP endpoints.

    Direct playback needs a backend that can presign against customer buckets and serve a
    byte-range playback manifest. Deployments without it can still ingest the MCAP normally.
    """


@dataclass(frozen=True)
class S3Source:
    """A customer-owned object referenced in place."""

    bucket: str
    key: str

    def to_conjure(self) -> Dict[str, Any]:
        return {"s3": {"bucket": self.bucket, "key": self.key}}


@dataclass(frozen=True)
class UriSource:
    """An https URL, used when the caller supplies a presigned or otherwise reachable location."""

    url: str

    def to_conjure(self) -> Dict[str, Any]:
        return {"uri": {"url": self.url}}


def source_from_uri(uri: str) -> "S3Source | UriSource":
    """Build the right source variant for a location the browser will read from."""
    if uri.startswith("s3://"):
        without_scheme = uri[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise ValueError(f"s3 uri must be of the form s3://bucket/key, got {uri!r}")
        return S3Source(bucket=bucket, key=key)
    if uri.startswith(("http://", "https://")):
        return UriSource(url=uri)
    raise ValueError(
        f"direct MCAP playback needs a location the browser can reach; {uri!r} is neither an s3:// "
        "path nor an https:// URL. Upload the file, or pass copy=True to ingest it."
    )


def _timestamp(nanoseconds: int) -> Dict[str, int]:
    seconds, nanos = divmod(nanoseconds, 1_000_000_000)
    return {"seconds": seconds, "nanos": nanos}


def _chunk_to_conjure(chunk: ChunkRange) -> Dict[str, Any]:
    return {
        "startTime": _timestamp(chunk.start_time),
        "endTime": _timestamp(chunk.end_time),
        "offset": chunk.offset,
        "length": chunk.length,
        "compression": chunk.compression,
        "compressedSize": chunk.compressed_size,
        "uncompressedSize": chunk.uncompressed_size,
    }


_MESSAGE_ENCODINGS = {
    MessageEncoding.PROTOBUF: "PROTOBUF",
    MessageEncoding.CDR: "CDR",
    MessageEncoding.UNKNOWN: "UNKNOWN",
}

_CODECS = {
    VideoCodec.H264: "H264",
    VideoCodec.H265: "H265",
    VideoCodec.VP9: "VP9",
    VideoCodec.AV1: "AV1",
    VideoCodec.UNKNOWN: "UNKNOWN",
}

_BITSTREAM_FORMATS = {
    BitstreamFormat.ANNEX_B: "ANNEX_B",
    BitstreamFormat.LENGTH_PREFIXED: "LENGTH_PREFIXED",
    BitstreamFormat.UNKNOWN: "UNKNOWN",
}

_CLASSIFICATIONS = {
    DirectPlaybackSupport.DIRECT: "DIRECT",
    DirectPlaybackSupport.NEEDS_PROCESSING: "NEEDS_PROCESSING",
    DirectPlaybackSupport.UNKNOWN: "UNKNOWN",
}


def channel_to_conjure(channel: McapVideoChannel, *, channel_name: str, tags: Mapping[str, str]) -> Dict[str, Any]:
    """Serialize one discovered video topic into the registration wire shape."""
    return {
        "channel": channel_name,
        "tags": dict(tags),
        "topic": channel.topic,
        "messageEncoding": _MESSAGE_ENCODINGS[channel.message_encoding],
        "codec": _CODECS[channel.codec],
        "bitstreamFormat": _BITSTREAM_FORMATS[channel.bitstream_format],
        "codecString": channel.codec_string,
        "width": channel.width,
        "height": channel.height,
        "frameRate": channel.frame_rate,
        "startTime": _timestamp(channel.start_time),
        "endTime": _timestamp(channel.end_time),
        "messageCount": channel.message_count,
        "keyframeIntervalSeconds": channel.keyframe_interval_seconds,
        "keyframeIntervalIsLowerBound": channel.keyframe_interval_is_lower_bound,
        "parameterSetsInBand": channel.parameter_sets_in_band,
        "hasBFrames": channel.has_b_frames,
        "monotonicTimestamps": channel.monotonic_timestamps,
        "captureVsLogSkewNanos": channel.capture_vs_log_skew_ns,
        "classification": _CLASSIFICATIONS[channel.support],
        "classificationReasons": list(channel.reasons),
        "chunks": [_chunk_to_conjure(chunk) for chunk in channel.chunk_ranges],
    }


@dataclass(frozen=True)
class DirectMcapFileRegistration:
    """One MCAP file and the video channels being registered from it."""

    source: "S3Source | UriSource"
    size_bytes: int
    channels: Sequence[Dict[str, Any]]
    etag: str | None = None
    mcap_library: str = ""
    mcap_profile: str = ""

    def to_conjure(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_conjure(),
            "sizeBytes": self.size_bytes,
            "etag": self.etag,
            "mcapLibrary": self.mcap_library,
            "mcapProfile": self.mcap_profile,
            "channels": list(self.channels),
        }


@dataclass(frozen=True)
class RegisteredDirectMcapChannel:
    """What the backend recorded for one registered channel."""

    channel: str
    topic: str
    series_rid: str | None = None
    video_file_rid: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def _from_conjure(cls, raw: Mapping[str, Any]) -> "RegisteredDirectMcapChannel":
        return cls(
            channel=str(raw.get("channel", "")),
            topic=str(raw.get("topic", "")),
            series_rid=raw.get("seriesRid"),
            video_file_rid=raw.get("videoFileRid"),
            tags=dict(raw.get("tags") or {}),
        )


class DirectMcapVideoService(Service):
    """Hand-written client for the direct-MCAP endpoints.

    Mirrors the transport conventions of the generated conjure services: an explicit
    ``auth_header`` first argument, JSON body, and JSON response.
    """

    def register(self, auth_header: str, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Register customer-owned MCAP video channels for direct playback."""
        return self._post(auth_header, DIRECT_MCAP_REGISTER_PATH, request)

    def get_playback_manifest(self, auth_header: str, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve a registered video channel to presigned URLs plus its cached chunk index."""
        return self._post(auth_header, DIRECT_MCAP_MANIFEST_PATH, request)

    def _post(self, auth_header: str, path: str, request: Mapping[str, Any]) -> Dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": auth_header,
        }
        try:
            response = self._request("POST", self._uri + path, params={}, headers=headers, json=dict(request))
        except ConjureHTTPError as exc:
            # A deployment predating the direct-MCAP endpoints answers 404/501 rather than a
            # conjure error, so translate that into advice the caller can act on.
            status = exc.response.status_code if exc.response is not None else None
            if status in (404, 501):
                raise DirectMcapUnsupportedError(
                    "this Nominal deployment does not support direct MCAP video playback "
                    f"(POST {path} returned {status}). Pass copy=True to ingest the file through the "
                    "standard video pipeline instead."
                ) from exc
            raise
        body = response.json() if response.content else None
        return dict(body) if body else {}


def build_register_request(
    *,
    dataset_rid: str,
    files: Sequence[DirectMcapFileRegistration],
) -> Dict[str, Any]:
    return {"datasetRid": dataset_rid, "files": [f.to_conjure() for f in files]}


def parse_register_response(raw: Mapping[str, Any]) -> List[RegisteredDirectMcapChannel]:
    return [RegisteredDirectMcapChannel._from_conjure(entry) for entry in raw.get("channels", []) or []]
