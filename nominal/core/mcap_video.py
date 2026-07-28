"""Register MCAP video topics with Nominal without copying or transcoding them.

Today, supporting a customer's video means copying the source into Nominal storage and re-encoding
it, roughly 1.5-2.5x the source footprint. An MCAP carries its own index, so a video topic inside one
can instead be registered in place: Nominal reads the file's summary section over a few ranged reads,
records the channels, time bounds and chunk index, and the browser later fetches video chunks
straight from the customer's bucket.

The entry point is :meth:`nominal.core.Dataset.add_mcap_video`, or equivalently
``Dataset.add_mcap(..., copy=False)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Sequence, TypeAlias

from nominal_api import api, ingest_api, scout_video_api, timeseries_metadata_api

from nominal.core._mcap.registration import (
    DirectMcapFileRegistration,
    RegisteredDirectMcapChannel,
    build_register_request,
    channel_to_conjure,
    parse_register_response,
    source_from_uri,
)
from nominal.core._mcap.sources import ByteSource, open_byte_source
from nominal.core._mcap.video import (
    DEFAULT_MAX_KEYFRAME_INTERVAL_SECONDS,
    DirectPlaybackSupport,
    McapVideoChannel,
    McapVideoInspection,
    inspect_mcap_video,
)
from nominal.core._utils.multipart import upload_multipart_file
from nominal.core.exceptions import NominalError
from nominal.core.filetype import FileTypes
from nominal.ts import _SecondsNanos

if TYPE_CHECKING:
    from nominal.core.dataset import Dataset

logger = logging.getLogger(__name__)

MCAP_SUFFIX = ".mcap"

UnsupportedVideoHandling: TypeAlias = Literal["error", "ingest", "skip"]
"""What to do with a video topic that cannot be played directly out of the customer's bucket."""


@dataclass(frozen=True)
class McapVideoChannelResult:
    """What happened to one video topic during registration."""

    topic: str
    channel: str
    """Name the topic was registered under in the series catalog."""

    inspection: McapVideoChannel
    source_uri: str
    registered: RegisteredDirectMcapChannel | None = None
    ingested: bool = False
    """True when the topic was routed to the copy-and-transcode pipeline instead."""

    skipped_reason: str | None = None

    @property
    def direct(self) -> bool:
        return self.registered is not None

    def describe(self) -> str:
        if self.direct:
            return f"{self.topic} -> direct playback as channel {self.channel!r}"
        if self.ingested:
            return f"{self.topic} -> ingested (copied and re-encoded): {'; '.join(self.inspection.reasons)}"
        return f"{self.topic} -> skipped: {self.skipped_reason or 'unsupported'}"


@dataclass(frozen=True)
class McapVideoRegistration:
    """Result of registering the video topics of one or more MCAP files."""

    dataset_rid: str
    files: Sequence[McapVideoInspection]
    channels: Sequence[McapVideoChannelResult]
    bytes_read: int = 0
    range_requests: int = 0

    @property
    def direct_channels(self) -> Sequence[McapVideoChannelResult]:
        return tuple(c for c in self.channels if c.direct)

    @property
    def ingested_channels(self) -> Sequence[McapVideoChannelResult]:
        return tuple(c for c in self.channels if c.ingested)

    @property
    def skipped_channels(self) -> Sequence[McapVideoChannelResult]:
        return tuple(c for c in self.channels if not c.direct and not c.ingested)

    def summary(self) -> str:
        """Human-readable account of what was registered, and at what cost."""
        source_bytes = sum(f.size_bytes for f in self.files)
        lines = [
            f"{len(self.files)} MCAP file(s), {len(self.channels)} video topic(s): "
            f"{len(self.direct_channels)} direct, {len(self.ingested_channels)} ingested, "
            f"{len(self.skipped_channels)} skipped",
            f"read {self.bytes_read} bytes in {self.range_requests} range request(s) "
            f"out of {source_bytes} bytes of source video",
        ]
        lines.extend(f"  {c.describe()}" for c in self.channels)
        return "\n".join(lines)


def default_channel_name(topic: str) -> str:
    """Channel name for a topic.

    The topic is used verbatim: an MCAP topic is already the name an engineer knows the camera by,
    and keeping it means the channel matches what they see in their own tooling.
    """
    return topic


def resolve_mcap_sources(uri: str, *, s3_client: Any = None) -> List[str]:
    """Expand a file or prefix into the list of MCAP objects to register.

    A prefix is listed with the caller's own credentials; nothing is downloaded.
    """
    if not uri.startswith("s3://"):
        return [uri]
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if key and not key.endswith("/"):
        return [uri]

    client = s3_client if s3_client is not None else _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    found: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for entry in page.get("Contents", []):
            if entry["Key"].endswith(MCAP_SUFFIX):
                found.append(f"s3://{bucket}/{entry['Key']}")
    if not found:
        raise FileNotFoundError(f"no {MCAP_SUFFIX} objects found under {uri}")
    return sorted(found)


def _s3_client() -> Any:
    from nominal.core._mcap.sources import McapSourceError

    try:
        import boto3
    except ImportError as exc:
        raise McapSourceError(
            "listing an s3:// prefix requires boto3 (pip install boto3); "
            "pass the full s3:// path to a single .mcap object instead"
        ) from exc
    return boto3.client("s3")


def inspect_sources(
    uris: Sequence[str],
    *,
    topics: Sequence[str] | None = None,
    **inspect_kwargs: object,
) -> List[McapVideoInspection]:
    """Inspect each MCAP, leaving the bytes where they are.

    The URI the caller asked for is what gets recorded, not whatever location the byte source
    happened to resolve to, because that URI is what playback will later be pointed at.
    """
    inspections: List[McapVideoInspection] = []
    for uri in uris:
        source: ByteSource = open_byte_source(uri)
        try:
            inspection = inspect_mcap_video(source, include_topics=topics, **inspect_kwargs)  # type: ignore[arg-type]
        finally:
            source.close()
        inspections.append(replace(inspection, uri=uri))
    return inspections


def build_file_registrations(
    inspections: Sequence[McapVideoInspection],
    *,
    tags: Mapping[str, str],
    channel_names: Mapping[str, str] | None = None,
) -> List[DirectMcapFileRegistration]:
    """Turn inspections into the payload the backend stores, skipping unplayable topics."""
    registrations: List[DirectMcapFileRegistration] = []
    for inspection in inspections:
        playable = [c for c in inspection.channels if c.support is DirectPlaybackSupport.DIRECT]
        if not playable:
            continue
        registrations.append(
            DirectMcapFileRegistration(
                source=source_from_uri(inspection.uri),
                size_bytes=inspection.size_bytes,
                etag=inspection.etag,
                mcap_library=inspection.library,
                mcap_profile=inspection.profile,
                channels=[
                    channel_to_conjure(
                        channel,
                        channel_name=(channel_names or {}).get(channel.topic, default_channel_name(channel.topic)),
                        tags=tags,
                    )
                    for channel in playable
                ],
            )
        )
    return registrations


def _register_mcap_video(
    dataset: "Dataset",
    source: str,
    *,
    topics: Sequence[str] | None,
    exclude_topics: Sequence[str] | None,
    tags: Mapping[str, str],
    copy: bool,
    on_unsupported: UnsupportedVideoHandling,
    channel_names: Mapping[str, str] | None,
    max_keyframe_interval_seconds: float,
) -> McapVideoRegistration:
    """Discover, classify and register the video topics of an MCAP source.

    Implements `Dataset.add_mcap_video`; kept out of `dataset.py` so the MCAP machinery stays in one
    place.
    """
    uris = resolve_mcap_sources(source)
    if not copy:
        _validate_remote_sources(uris)

    inspections = inspect_sources(
        uris,
        topics=topics,
        max_keyframe_interval_seconds=max_keyframe_interval_seconds,
    )
    if exclude_topics:
        excluded = set(exclude_topics)
        inspections = [
            replace(inspection, channels=tuple(c for c in inspection.channels if c.topic not in excluded))
            for inspection in inspections
        ]

    bytes_read = sum(i.bytes_read for i in inspections)
    range_requests = sum(i.range_requests for i in inspections)
    resolved_names = dict(channel_names or {})

    if copy:
        return McapVideoRegistration(
            dataset_rid=dataset.rid,
            files=tuple(inspections),
            channels=tuple(_ingest_all(dataset, inspections, tags=tags, channel_names=resolved_names)),
            bytes_read=bytes_read,
            range_requests=range_requests,
        )

    unsupported = [c for i in inspections for c in i.channels if c.support is not DirectPlaybackSupport.DIRECT]
    if unsupported and on_unsupported == "error":
        detail = "; ".join(f"{c.topic}: {', '.join(c.reasons) or 'unsupported'}" for c in unsupported)
        raise NominalError(
            f"{len(unsupported)} video topic(s) cannot be played directly from their source: {detail}. "
            'Pass on_unsupported="ingest" to copy and re-encode just those topics, or '
            '"skip" to leave them unregistered.'
        )

    # Local files are uploaded only now: after inspection, so an unplayable recording fails
    # before the bytes move, and only for files that actually have a playable topic.
    inspections = [_upload_if_local(dataset, inspection) for inspection in inspections]

    registrations = build_file_registrations(inspections, tags=tags, channel_names=resolved_names)
    registered: Dict[str, RegisteredDirectMcapChannel] = {}
    if registrations:
        _create_video_series(dataset, inspections, tags=tags, channel_names=resolved_names)
        response = dataset._clients.direct_mcap.register(
            dataset._clients.auth_header,
            build_register_request(dataset_rid=dataset.rid, files=registrations),
        )
        for entry in parse_register_response(response):
            registered[entry.topic] = entry

    results: List[McapVideoChannelResult] = []
    for inspection in inspections:
        for channel in inspection.channels:
            name = resolved_names.get(channel.topic, default_channel_name(channel.topic))
            if channel.support is DirectPlaybackSupport.DIRECT:
                results.append(
                    McapVideoChannelResult(
                        topic=channel.topic,
                        channel=name,
                        inspection=channel,
                        source_uri=inspection.uri,
                        registered=registered.get(channel.topic),
                    )
                )
            elif on_unsupported == "ingest":
                results.append(_ingest_topic(dataset, inspection, channel, name, tags))
            else:
                results.append(
                    McapVideoChannelResult(
                        topic=channel.topic,
                        channel=name,
                        inspection=channel,
                        source_uri=inspection.uri,
                        skipped_reason="; ".join(channel.reasons) or "unsupported for direct playback",
                    )
                )

    return McapVideoRegistration(
        dataset_rid=dataset.rid,
        files=tuple(inspections),
        channels=tuple(results),
        bytes_read=bytes_read,
        range_requests=range_requests,
    )


def _create_video_series(
    dataset: "Dataset",
    inspections: Sequence[McapVideoInspection],
    *,
    tags: Mapping[str, str],
    channel_names: Mapping[str, str],
) -> None:
    """Register the discovered topics as VIDEO channels in the series catalog.

    This is the existing, idempotent series-archetype call: an MCAP-backed channel is an ordinary
    video channel as far as search, pickers and workbooks are concerned.
    """
    requests = [
        timeseries_metadata_api.CreateVideoSeriesRequest(
            dataset_rid=dataset.rid,
            channel=channel_names.get(channel.topic, default_channel_name(channel.topic)),
            tags=dict(tags),
            time_bounds=timeseries_metadata_api.TimeBounds(
                start=_SecondsNanos.from_nanoseconds(channel.start_time).to_api(),
                end=_SecondsNanos.from_nanoseconds(channel.end_time).to_api(),
            ),
        )
        for inspection in inspections
        for channel in inspection.channels
        if channel.support is DirectPlaybackSupport.DIRECT
    ]
    if not requests:
        return
    dataset._clients.series_metadata.batch_create_video_series(
        dataset._clients.auth_header,
        timeseries_metadata_api.BatchCreateVideoSeriesRequest(requests=requests),
    )


def _ingest_all(
    dataset: "Dataset",
    inspections: Sequence[McapVideoInspection],
    *,
    tags: Mapping[str, str],
    channel_names: Mapping[str, str],
) -> List[McapVideoChannelResult]:
    results: List[McapVideoChannelResult] = []
    for inspection in inspections:
        for channel in inspection.channels:
            name = channel_names.get(channel.topic, default_channel_name(channel.topic))
            results.append(_ingest_topic(dataset, inspection, channel, name, tags))
    return results


def _ingest_topic(
    dataset: "Dataset",
    inspection: McapVideoInspection,
    channel: McapVideoChannel,
    channel_name: str,
    tags: Mapping[str, str],
) -> McapVideoChannelResult:
    """Route one topic through the copy-and-transcode pipeline."""
    logger.info(
        "ingesting MCAP video topic %s from %s (%s)",
        channel.topic,
        inspection.uri,
        "; ".join(channel.reasons) or "requested",
    )
    ingest_source = _ingest_source_for(dataset, inspection.uri)
    request = ingest_api.IngestRequest(
        options=ingest_api.IngestOptions(
            video_v2=ingest_api.VideoOptsV2(
                source=ingest_source,
                target=ingest_api.DatasetIngestTarget(
                    existing=ingest_api.ExistingDatasetIngestDestination(dataset_rid=dataset.rid)
                ),
                timestamp_manifest=scout_video_api.VideoFileTimestampManifest(
                    mcap=scout_video_api.McapTimestampManifest(api.McapChannelLocator(topic=channel.topic))
                ),
                channel=channel_name,
                tags=dict(tags),
            )
        )
    )
    dataset._clients.ingest.ingest(dataset._clients.auth_header, request)
    return McapVideoChannelResult(
        topic=channel.topic,
        channel=channel_name,
        inspection=channel,
        source_uri=inspection.uri,
        ingested=True,
    )


def _is_local_path(uri: str) -> bool:
    return not uri.startswith(("s3://", "http://", "https://"))


def _validate_remote_sources(uris: Sequence[str]) -> None:
    """Fail on a malformed remote source before doing any work, not after inspecting gigabytes.

    Local paths are exempt: a browser cannot range-read them, so the bytes are uploaded into
    Nominal storage after inspection and the upload target is registered instead.
    """
    for uri in uris:
        if not _is_local_path(uri):
            source_from_uri(uri)


def _upload_if_local(dataset: "Dataset", inspection: McapVideoInspection) -> McapVideoInspection:
    """Upload a locally-inspected MCAP into Nominal storage and point its registration there.

    A local path is inspected directly, which is byte-identical to and cheaper than inspecting
    the uploaded copy. The registration, though, must name a location a browser can range-read,
    so the file goes through the standard multipart upload and the inspection is rewritten to
    the uploaded object. This stores the bytes in Nominal, which the no-copy path exists to
    avoid: it is a dev and test convenience, and the real product flow registers files that
    already live in customer storage.
    """
    if not _is_local_path(inspection.uri):
        return inspection
    if not any(c.support is DirectPlaybackSupport.DIRECT for c in inspection.channels):
        return inspection
    s3_path = upload_multipart_file(
        dataset._clients.auth_header,
        dataset._clients.resolve_default_workspace_rid(),
        Path(inspection.uri),
        dataset._clients.upload,
        file_type=FileTypes.MCAP,
        header_provider=dataset._clients.header_provider,
    )
    # The uploaded object's etag is the multipart composite, not the md5 of the local bytes;
    # registering no etag is honest, while registering the local hash would be wrong.
    return replace(inspection, uri=s3_path, etag=None)


def _ingest_source_for(dataset: "Dataset", uri: str) -> ingest_api.IngestSource:
    """Give the ingest pipeline a location it can read, uploading a local file if need be."""
    if uri.startswith("s3://"):
        return ingest_api.IngestSource(s3=ingest_api.S3IngestSource(path=uri))
    if uri.startswith(("http://", "https://")):
        return ingest_api.IngestSource(presigned_file=ingest_api.PresignedFileIngestSource(url=uri))

    path = Path(uri)
    s3_path = upload_multipart_file(
        dataset._clients.auth_header,
        dataset._clients.resolve_default_workspace_rid(),
        path,
        dataset._clients.upload,
        file_type=FileTypes.MCAP,
        header_provider=dataset._clients.header_provider,
    )
    return ingest_api.IngestSource(s3=ingest_api.S3IngestSource(path=s3_path))


__all__ = [
    "DEFAULT_MAX_KEYFRAME_INTERVAL_SECONDS",
    "McapVideoChannelResult",
    "McapVideoRegistration",
    "UnsupportedVideoHandling",
    "build_file_registrations",
    "build_register_request",
    "default_channel_name",
    "inspect_sources",
    "parse_register_response",
    "resolve_mcap_sources",
]
