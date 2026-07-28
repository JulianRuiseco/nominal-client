"""Discover video topics in an MCAP and decide whether a browser can play them in place.

Registration is deliberately cheap: read the file's own summary section for channels and time
bounds, then probe one chunk per video channel to learn the codec, the bitstream shape, whether
parameter sets are in band, and how far apart true keyframes are. Everything here is a ranged read
of a few hundred kilobytes, whatever the size of the recording.
"""

from __future__ import annotations

import enum
import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

from nominal.core._mcap.bitstream import (
    AccessUnitScan,
    BitstreamFormat,
    FrameKind,
    VideoCodec,
    scan_access_unit,
)
from nominal.core._mcap.messages import (
    CompressedVideoMessage,
    MessageDecodeError,
    decode_cdr_compressed_video,
    decode_protobuf_compressed_video,
)
from nominal.core._mcap.reader import (
    McapSummary,
    decompress_chunk,
    iter_chunk_messages,
    read_chunk,
    read_summary,
)
from nominal.core._mcap.records import ChannelRecord, ChunkIndexRecord, MessageRecord, SchemaRecord
from nominal.core._mcap.sources import ByteSource, McapSourceError, open_byte_source

logger = logging.getLogger(__name__)

PROTOBUF_VIDEO_SCHEMAS = frozenset({"foxglove.CompressedVideo"})
ROS_VIDEO_SCHEMAS = frozenset({"foxglove_msgs/msg/CompressedVideo", "foxglove_msgs/CompressedVideo"})

DEFAULT_MAX_KEYFRAME_INTERVAL_SECONDS = 10.0
"""Above this, a seek decodes forward too long for the panel to feel responsive."""

DEFAULT_MAX_CHUNK_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
"""The playback unit is one decompressed chunk, so a browser has to hold one in memory."""

DEFAULT_MAX_PROBE_FRAMES = 400
"""Frames decoded per probed chunk. Enough to see several keyframe intervals."""

DEFAULT_PROBE_CHUNKS = 2
"""Chunks probed per channel: the first (parameter sets, codec) and the last (cadence drift)."""

BROWSER_DECODABLE_CODECS = frozenset({VideoCodec.H264, VideoCodec.H265})
"""Codecs WebCodecs can decode. H.265 support is hardware-dependent, flagged separately."""


class MessageEncoding(str, enum.Enum):
    """Wire envelope carrying the video frames."""

    PROTOBUF = "protobuf"
    CDR = "cdr"
    UNKNOWN = "unknown"


class DirectPlaybackSupport(str, enum.Enum):
    """Whether a channel can be played straight out of the customer's bucket."""

    DIRECT = "direct"
    """Range-read and decode in the browser. No copy, no transcode."""

    NEEDS_PROCESSING = "needs_processing"
    """Route to the ingest pipeline: unsupported codec, sparse keyframes, or an unreadable layout."""

    UNKNOWN = "unknown"
    """The probe could not run (for example a missing chunk decompressor). Treated as unsupported."""


@dataclass(frozen=True)
class ChunkRange:
    """One entry of the file's own chunk index, in the shape playback needs."""

    start_time: int
    end_time: int
    offset: int
    length: int
    compression: str
    compressed_size: int
    uncompressed_size: int

    @classmethod
    def from_index(cls, index: ChunkIndexRecord) -> "ChunkRange":
        return cls(
            start_time=index.message_start_time,
            end_time=index.message_end_time,
            offset=index.chunk_start_offset,
            length=index.chunk_length,
            compression=index.compression,
            compressed_size=index.compressed_size,
            uncompressed_size=index.uncompressed_size,
        )


@dataclass(frozen=True)
class McapVideoChannel:
    """A video topic discovered inside an MCAP, with its direct-playback verdict."""

    topic: str
    channel_id: int
    schema_name: str
    message_encoding: MessageEncoding
    codec: VideoCodec
    bitstream_format: BitstreamFormat
    start_time: int
    end_time: int
    support: DirectPlaybackSupport
    reasons: Sequence[str] = field(default_factory=tuple)
    message_count: int | None = None
    width: int | None = None
    height: int | None = None
    codec_string: str | None = None
    frame_rate: float | None = None
    keyframe_interval_seconds: float | None = None
    keyframe_interval_is_lower_bound: bool = False
    """True when the probe saw fewer than two keyframes, so the interval could be longer."""

    parameter_sets_in_band: bool | None = None
    has_b_frames: bool | None = None
    frames_probed: int = 0
    keyframes_probed: int = 0
    intra_non_idr_probed: int = 0
    max_chunk_uncompressed_bytes: int = 0
    chunk_count: int = 0
    capture_vs_log_skew_ns: int | None = None
    monotonic_timestamps: bool = True
    chunk_ranges: Sequence[ChunkRange] = field(default_factory=tuple, repr=False)

    @property
    def is_direct_playable(self) -> bool:
        return self.support == DirectPlaybackSupport.DIRECT

    @property
    def duration_seconds(self) -> float:
        return max(self.end_time - self.start_time, 0) / 1e9

    def describe(self) -> str:
        """One-line human summary, used by CLI output and log messages."""
        size = f"{self.width}x{self.height}" if self.width and self.height else "unknown size"
        verdict = self.support.value
        detail = f"; {'; '.join(self.reasons)}" if self.reasons else ""
        return f"{self.topic} [{self.codec.value} {size}, {self.duration_seconds:.1f}s] -> {verdict}{detail}"


@dataclass(frozen=True)
class McapVideoInspection:
    """Result of inspecting one MCAP file for video topics."""

    uri: str
    size_bytes: int
    etag: str | None
    profile: str
    library: str
    is_chunked: bool
    start_time: int | None
    end_time: int | None
    channels: Sequence[McapVideoChannel]
    non_video_topics: Sequence[str] = field(default_factory=tuple)
    bytes_read: int = 0
    range_requests: int = 0

    @property
    def direct_playable_channels(self) -> Sequence[McapVideoChannel]:
        return tuple(c for c in self.channels if c.is_direct_playable)

    @property
    def needs_processing_channels(self) -> Sequence[McapVideoChannel]:
        return tuple(c for c in self.channels if not c.is_direct_playable)

    def channel(self, topic: str) -> McapVideoChannel:
        for candidate in self.channels:
            if candidate.topic == topic:
                return candidate
        known = ", ".join(c.topic for c in self.channels) or "none"
        raise KeyError(f"no video topic {topic!r} in {self.uri} (found: {known})")

    def summary_line(self) -> str:
        direct = len(self.direct_playable_channels)
        return (
            f"{self.uri}: {len(self.channels)} video topic(s), {direct} direct-playable, "
            f"read {self.bytes_read} bytes in {self.range_requests} range request(s)"
        )


def inspect_mcap_video(
    source: ByteSource | str,
    *,
    include_topics: Sequence[str] | None = None,
    probe: bool = True,
    probe_chunks: int = DEFAULT_PROBE_CHUNKS,
    max_probe_frames: int = DEFAULT_MAX_PROBE_FRAMES,
    max_keyframe_interval_seconds: float = DEFAULT_MAX_KEYFRAME_INTERVAL_SECONDS,
    max_chunk_uncompressed_bytes: int = DEFAULT_MAX_CHUNK_UNCOMPRESSED_BYTES,
    expected_etag: str | None = None,
) -> McapVideoInspection:
    """Discover video topics in an MCAP and classify each for direct playback.

    Args:
        source: An open :class:`ByteSource`, or a local path / ``https://`` / ``s3://`` URI.
        include_topics: Restrict inspection to these topics. Defaults to every video topic found.
        probe: Read one chunk per channel to determine codec, bitstream shape and keyframe cadence.
            With ``probe=False`` only the summary section is read and every channel is classified
            ``UNKNOWN``.
        probe_chunks: How many chunks to read per channel during the probe.
        max_probe_frames: Frames to scan per probed chunk.
        max_keyframe_interval_seconds: Longest keyframe spacing still considered seekable.
        max_chunk_uncompressed_bytes: Largest decompressed chunk a browser is asked to hold.
        expected_etag: Fail if the object's entity tag differs, so an overwritten recording is
            caught rather than silently played.

    Returns:
        The channels found and, for each, whether it can be played without copying or transcoding.
    """
    owns_source = isinstance(source, str)
    byte_source = open_byte_source(source, expected_etag=expected_etag) if isinstance(source, str) else source
    try:
        summary = read_summary(byte_source)
        return _inspect(
            byte_source,
            summary,
            include_topics=include_topics,
            probe=probe,
            probe_chunks=probe_chunks,
            max_probe_frames=max_probe_frames,
            max_keyframe_interval_seconds=max_keyframe_interval_seconds,
            max_chunk_uncompressed_bytes=max_chunk_uncompressed_bytes,
        )
    finally:
        if owns_source:
            byte_source.close()


def _inspect(
    source: ByteSource,
    summary: McapSummary,
    *,
    include_topics: Sequence[str] | None,
    probe: bool,
    probe_chunks: int,
    max_probe_frames: int,
    max_keyframe_interval_seconds: float,
    max_chunk_uncompressed_bytes: int,
) -> McapVideoInspection:
    wanted = set(include_topics) if include_topics else None
    channels: List[McapVideoChannel] = []
    non_video: List[str] = []

    for channel in summary.channels.values():
        schema = summary.schemas.get(channel.schema_id)
        encoding = _classify_encoding(channel, schema)
        if encoding is MessageEncoding.UNKNOWN:
            non_video.append(channel.topic)
            continue
        if wanted is not None and channel.topic not in wanted:
            continue
        channels.append(
            _build_channel(
                source,
                summary,
                channel,
                schema,
                encoding,
                probe=probe and summary.is_chunked,
                probe_chunks=probe_chunks,
                max_probe_frames=max_probe_frames,
                max_keyframe_interval_seconds=max_keyframe_interval_seconds,
                max_chunk_uncompressed_bytes=max_chunk_uncompressed_bytes,
            )
        )

    if wanted is not None:
        missing = wanted - {c.topic for c in channels}
        if missing:
            found = ", ".join(sorted(c.topic for c in channels)) or "none"
            raise KeyError(f"topics not found as video channels in {source.uri}: {sorted(missing)} (found: {found})")

    channels.sort(key=lambda c: c.topic)
    return McapVideoInspection(
        uri=source.uri,
        size_bytes=source.size,
        etag=source.etag,
        profile=summary.profile,
        library=summary.library,
        is_chunked=summary.is_chunked,
        start_time=summary.message_start_time,
        end_time=summary.message_end_time,
        channels=tuple(channels),
        non_video_topics=tuple(sorted(non_video)),
        bytes_read=getattr(source, "bytes_read", 0),
        range_requests=getattr(source, "range_requests", 0),
    )


def _classify_encoding(channel: ChannelRecord, schema: SchemaRecord | None) -> MessageEncoding:
    schema_name = schema.name if schema is not None else ""
    message_encoding = channel.message_encoding.lower()
    if schema_name in PROTOBUF_VIDEO_SCHEMAS:
        return MessageEncoding.PROTOBUF if message_encoding in ("protobuf", "", "proto") else MessageEncoding.UNKNOWN
    if schema_name in ROS_VIDEO_SCHEMAS:
        return MessageEncoding.CDR if message_encoding in ("cdr", "ros2", "") else MessageEncoding.UNKNOWN
    return MessageEncoding.UNKNOWN


def _decode(encoding: MessageEncoding, data: bytes) -> CompressedVideoMessage:
    if encoding is MessageEncoding.PROTOBUF:
        return decode_protobuf_compressed_video(data)
    return decode_cdr_compressed_video(data)


def _build_channel(
    source: ByteSource,
    summary: McapSummary,
    channel: ChannelRecord,
    schema: SchemaRecord | None,
    encoding: MessageEncoding,
    *,
    probe: bool,
    probe_chunks: int,
    max_probe_frames: int,
    max_keyframe_interval_seconds: float,
    max_chunk_uncompressed_bytes: int,
) -> McapVideoChannel:
    bounds = summary.channel_time_bounds(channel.id) or (0, 0)
    chunks = summary.chunks_for_channel(channel.id)
    chunk_ranges = tuple(ChunkRange.from_index(c) for c in chunks)
    max_chunk_bytes = max((c.uncompressed_size for c in chunks), default=0)

    base = dict(
        topic=channel.topic,
        channel_id=channel.id,
        schema_name=schema.name if schema is not None else "",
        message_encoding=encoding,
        start_time=bounds[0],
        end_time=bounds[1],
        message_count=summary.message_count(channel.id),
        max_chunk_uncompressed_bytes=max_chunk_bytes,
        chunk_count=len(chunks),
        chunk_ranges=chunk_ranges,
    )

    reasons: List[str] = []
    if not summary.is_chunked:
        reasons.append(
            "file has no chunk index, so it cannot be range-read "
            "(truncated recording, or a writer preset such as rosbag2 'fastwrite')"
        )
    if not probe:
        if summary.is_chunked:
            reasons.append("codec probe was skipped")
        return McapVideoChannel(
            codec=VideoCodec.UNKNOWN,
            bitstream_format=BitstreamFormat.UNKNOWN,
            support=(
                DirectPlaybackSupport.NEEDS_PROCESSING if not summary.is_chunked else DirectPlaybackSupport.UNKNOWN
            ),
            reasons=tuple(reasons),
            **base,  # type: ignore[arg-type]
        )

    try:
        probed = _probe_channel(source, channel, encoding, chunks, probe_chunks, max_probe_frames)
    except (McapSourceError, MessageDecodeError) as exc:
        logger.warning("could not probe MCAP video topic %s: %s", channel.topic, exc)
        reasons.append(f"probe failed: {exc}")
        return McapVideoChannel(
            codec=VideoCodec.UNKNOWN,
            bitstream_format=BitstreamFormat.UNKNOWN,
            support=DirectPlaybackSupport.UNKNOWN,
            reasons=tuple(reasons),
            **base,  # type: ignore[arg-type]
        )

    support, verdict_reasons = _classify(
        probed,
        max_keyframe_interval_seconds=max_keyframe_interval_seconds,
        max_chunk_uncompressed_bytes=max_chunk_uncompressed_bytes,
        max_chunk_bytes=max_chunk_bytes,
        chunked=summary.is_chunked,
    )
    reasons.extend(verdict_reasons)

    return McapVideoChannel(
        codec=probed.codec,
        bitstream_format=probed.bitstream_format,
        support=support,
        reasons=tuple(reasons),
        width=probed.width,
        height=probed.height,
        codec_string=probed.codec_string,
        frame_rate=probed.frame_rate,
        keyframe_interval_seconds=probed.keyframe_interval_seconds,
        keyframe_interval_is_lower_bound=probed.keyframe_interval_is_lower_bound,
        parameter_sets_in_band=probed.parameter_sets_in_band,
        has_b_frames=probed.has_b_frames,
        frames_probed=probed.frames,
        keyframes_probed=probed.keyframes,
        intra_non_idr_probed=probed.intra_non_idr,
        capture_vs_log_skew_ns=probed.capture_vs_log_skew_ns,
        monotonic_timestamps=probed.monotonic,
        **base,  # type: ignore[arg-type]
    )


@dataclass
class _ProbeResult:
    codec: VideoCodec = VideoCodec.UNKNOWN
    bitstream_format: BitstreamFormat = BitstreamFormat.UNKNOWN
    width: int | None = None
    height: int | None = None
    codec_string: str | None = None
    frame_rate: float | None = None
    keyframe_interval_seconds: float | None = None
    keyframe_interval_is_lower_bound: bool = False
    parameter_sets_in_band: bool | None = None
    has_b_frames: bool | None = None
    frames: int = 0
    keyframes: int = 0
    intra_non_idr: int = 0
    capture_vs_log_skew_ns: int | None = None
    monotonic: bool = True


def _probe_channel(
    source: ByteSource,
    channel: ChannelRecord,
    encoding: MessageEncoding,
    chunks: Sequence[ChunkIndexRecord],
    probe_chunks: int,
    max_probe_frames: int,
) -> _ProbeResult:
    """Read the first and last chunks holding this channel and scan their frames."""
    result = _ProbeResult()
    if not chunks:
        return result

    tally = _ProbeTally()
    for index in _select_probe_chunks(chunks, probe_chunks):
        # Each probed chunk is its own window. Probed chunks are deliberately not adjacent, so
        # nothing may be measured across the gap between them.
        tally.start_window()
        body = decompress_chunk(read_chunk(source, index))
        for message in iter_chunk_messages(body, channel.id)[:max_probe_frames]:
            _probe_message(result, tally, message, encoding)

    if result.keyframes:
        result.parameter_sets_in_band = tally.keyframes_missing_parameter_sets == 0
    result.frame_rate = _estimate_rate(tally.windows)
    result.keyframe_interval_seconds, result.keyframe_interval_is_lower_bound = _estimate_keyframe_interval(
        tally.windows
    )
    if tally.skews:
        result.capture_vs_log_skew_ns = int(statistics.median(tally.skews))
    return result


@dataclass
class _ProbeWindow:
    """Frames observed within one probed chunk, which is one contiguous run of video."""

    frame_times: List[int] = field(default_factory=list)
    keyframe_times: List[int] = field(default_factory=list)


@dataclass
class _ProbeTally:
    """Per-frame observations accumulated while scanning probed chunks.

    Timings are kept per window because the probe reads chunks from opposite ends of the file.
    Measuring a keyframe gap across that hole would report the distance between the first and last
    chunk rather than the encoder's actual cadence, condemning well-formed files to re-encoding.
    """

    windows: List[_ProbeWindow] = field(default_factory=list)
    skews: List[int] = field(default_factory=list)
    keyframes_missing_parameter_sets: int = 0
    last_log_time: int | None = None

    def start_window(self) -> None:
        self.windows.append(_ProbeWindow())

    @property
    def current(self) -> _ProbeWindow:
        if not self.windows:
            self.start_window()
        return self.windows[-1]


def _probe_message(
    result: _ProbeResult,
    tally: _ProbeTally,
    message: MessageRecord,
    encoding: MessageEncoding,
) -> None:
    """Fold one ``CompressedVideo`` message into the running probe state."""
    try:
        decoded = _decode(encoding, message.data)
    except MessageDecodeError:
        return

    if result.codec is VideoCodec.UNKNOWN and decoded.format:
        result.codec = VideoCodec.parse(decoded.format)

    scan = scan_access_unit(decoded.data, result.codec, result.bitstream_format)
    if result.bitstream_format is BitstreamFormat.UNKNOWN:
        result.bitstream_format = scan.format

    result.frames += 1
    window = tally.current
    window.frame_times.append(message.log_time)
    if tally.last_log_time is not None and message.log_time < tally.last_log_time:
        result.monotonic = False
    tally.last_log_time = message.log_time
    if decoded.timestamp_ns is not None:
        tally.skews.append(message.log_time - decoded.timestamp_ns)

    _accumulate_scan(result, scan)
    if scan.kind is FrameKind.KEYFRAME:
        result.keyframes += 1
        window.keyframe_times.append(message.log_time)
        if not (scan.has_sps and scan.has_pps):
            tally.keyframes_missing_parameter_sets += 1
    elif scan.kind is FrameKind.INTRA_NON_IDR:
        result.intra_non_idr += 1


def _accumulate_scan(result: _ProbeResult, scan: AccessUnitScan) -> None:
    if scan.sps is not None and result.width is None:
        result.width = scan.sps.width
        result.height = scan.sps.height
        result.codec_string = scan.sps.codec_string
    if scan.has_b_slices:
        result.has_b_frames = True
    elif result.has_b_frames is None:
        result.has_b_frames = False


def _select_probe_chunks(chunks: Sequence[ChunkIndexRecord], probe_chunks: int) -> Sequence[ChunkIndexRecord]:
    if probe_chunks <= 0:
        return ()
    if len(chunks) <= probe_chunks:
        return chunks
    if probe_chunks == 1:
        return (chunks[0],)
    # First chunk shows how a decoder starts; last chunk shows whether cadence holds to the end.
    step = (len(chunks) - 1) / (probe_chunks - 1)
    picked = {int(round(i * step)) for i in range(probe_chunks)}
    return tuple(chunks[i] for i in sorted(picked))


def _estimate_rate(windows: Sequence["_ProbeWindow"]) -> float | None:
    """Frame rate from within-window frame spacing only."""
    deltas = [b - a for window in windows for a, b in zip(window.frame_times, window.frame_times[1:]) if b > a]
    if not deltas:
        return None
    median_delta = statistics.median(deltas)
    if median_delta <= 0:
        return None
    return round(1e9 / median_delta, 3)


def _estimate_keyframe_interval(windows: Sequence["_ProbeWindow"]) -> tuple[float | None, bool]:
    """Return ``(interval_seconds, is_lower_bound)`` measured within contiguous windows.

    Two or more keyframes inside one window give the encoder's real worst-case gap. A window with
    fewer than two only tells us the interval is at least as long as that window, which is a lower
    bound: enough to reject a file that already exceeds the threshold, never enough to clear one.
    """
    measured = [b - a for window in windows for a, b in zip(window.keyframe_times, window.keyframe_times[1:]) if b > a]
    if measured:
        return round(max(measured) / 1e9, 3), False

    spans: List[float] = []
    for window in windows:
        if not window.frame_times:
            continue
        start = window.keyframe_times[0] if window.keyframe_times else min(window.frame_times)
        span = (max(window.frame_times) - start) / 1e9
        if span > 0:
            spans.append(span)
    if not spans:
        return None, True
    return round(max(spans), 3), True


def _classify(
    probed: _ProbeResult,
    *,
    max_keyframe_interval_seconds: float,
    max_chunk_uncompressed_bytes: int,
    max_chunk_bytes: int,
    chunked: bool,
) -> tuple[DirectPlaybackSupport, List[str]]:
    reasons: List[str] = []

    if not chunked:
        return DirectPlaybackSupport.NEEDS_PROCESSING, reasons

    if probed.frames == 0:
        reasons.append("no decodable CompressedVideo messages found in the probed chunks")
        return DirectPlaybackSupport.UNKNOWN, reasons

    blocking = False
    for check in (
        _check_codec,
        _check_bitstream,
        _check_keyframes,
        _check_parameter_sets,
        _check_timestamps,
        _check_chunk_size,
    ):
        blocks, messages = check(
            probed,
            max_keyframe_interval_seconds=max_keyframe_interval_seconds,
            max_chunk_uncompressed_bytes=max_chunk_uncompressed_bytes,
            max_chunk_bytes=max_chunk_bytes,
        )
        blocking = blocking or blocks
        reasons.extend(messages)

    if blocking:
        return DirectPlaybackSupport.NEEDS_PROCESSING, reasons
    return DirectPlaybackSupport.DIRECT, reasons


def _check_codec(probed: _ProbeResult, **_: float) -> tuple[bool, List[str]]:
    if probed.codec not in BROWSER_DECODABLE_CODECS:
        return True, [f"codec {probed.codec.value} cannot be decoded by WebCodecs"]
    if probed.codec is VideoCodec.H265:
        return False, [
            "H.265 decode in the browser is hardware-dependent (Chrome ships no software fallback), "
            "so playback will fail on machines without HEVC support"
        ]
    return False, []


def _check_bitstream(probed: _ProbeResult, **_: float) -> tuple[bool, List[str]]:
    messages: List[str] = []
    blocking = False
    if probed.bitstream_format is BitstreamFormat.UNKNOWN:
        messages.append("could not determine whether the bitstream is Annex B or length-prefixed")
        blocking = True
    if probed.has_b_frames:
        messages.append("bitstream contains B-frames, which the direct player does not reorder")
        blocking = True
    return blocking, messages


def _check_keyframes(
    probed: _ProbeResult, *, max_keyframe_interval_seconds: float = 0.0, **_: float
) -> tuple[bool, List[str]]:
    if probed.keyframes == 0:
        detail = ""
        if probed.intra_non_idr:
            detail = (
                f" ({probed.intra_non_idr} non-IDR I-frame(s) were seen; these are not seek points "
                "- some hardware encoders emit them between sparse IDRs)"
            )
        return True, [f"no IDR keyframe found in {probed.frames} probed frame(s){detail}"]

    interval = probed.keyframe_interval_seconds
    if interval is None:
        return False, []
    if interval > max_keyframe_interval_seconds:
        qualifier = "at least " if probed.keyframe_interval_is_lower_bound else ""
        return True, [
            f"keyframes are {qualifier}{interval:.1f}s apart, above the "
            f"{max_keyframe_interval_seconds:.1f}s seek threshold"
        ]
    if probed.keyframe_interval_is_lower_bound:
        # Only one keyframe in the probe window: the cadence could be worse than it looks, but
        # nothing observed disqualifies the file.
        return False, [
            f"only {probed.keyframes} keyframe(s) in the {interval:.1f}s probed, so the keyframe "
            "interval is a lower bound rather than a measurement"
        ]
    return False, []


def _check_parameter_sets(probed: _ProbeResult, **_: float) -> tuple[bool, List[str]]:
    if probed.parameter_sets_in_band is False:
        return False, [
            "keyframes do not carry SPS/PPS in band; the player caches parameter sets from the "
            "start of the stream and injects them before such keyframes"
        ]
    return False, []


def _check_timestamps(probed: _ProbeResult, **_: float) -> tuple[bool, List[str]]:
    if not probed.monotonic:
        return False, ["message timestamps are not monotonic; the player clamps them during playback"]
    return False, []


def _check_chunk_size(
    probed: _ProbeResult,
    *,
    max_chunk_uncompressed_bytes: float = 0.0,
    max_chunk_bytes: float = 0.0,
    **_: float,
) -> tuple[bool, List[str]]:
    if max_chunk_bytes > max_chunk_uncompressed_bytes:
        return True, [
            f"largest chunk decompresses to {max_chunk_bytes / 1024 / 1024:.1f} MiB, above the "
            f"{max_chunk_uncompressed_bytes / 1024 / 1024:.1f} MiB playback limit"
        ]
    return False, []


def video_topics(source: ByteSource | str, **kwargs: object) -> Mapping[str, McapVideoChannel]:
    """Convenience wrapper returning discovered video channels keyed by topic."""
    inspection = inspect_mcap_video(source, **kwargs)  # type: ignore[arg-type]
    return {channel.topic: channel for channel in inspection.channels}
