"""Read an MCAP's own summary section with a handful of ranged reads.

The summary section sits between ``summary_start`` (named in the footer) and the footer itself. It
repeats every schema and channel record, and adds the chunk index and statistics. Reading it costs
two ranged reads for a typical file, regardless of how large the recording is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from nominal.core._mcap.records import (
    MAGIC,
    TAIL_SIZE,
    ChannelRecord,
    ChunkIndexRecord,
    ChunkRecord,
    McapParseError,
    MessageRecord,
    Opcode,
    SchemaRecord,
    StatisticsRecord,
    iter_records,
    parse_channel,
    parse_chunk,
    parse_chunk_index,
    parse_footer,
    parse_message,
    parse_schema,
    parse_statistics,
)
from nominal.core._mcap.sources import ByteSource, McapSourceError

logger = logging.getLogger(__name__)

MAX_SUMMARY_BYTES = 256 * 1024 * 1024
"""Refuse to buffer an implausibly large summary section rather than exhausting memory."""


@dataclass(frozen=True)
class McapSummary:
    """Everything registration needs, read from the tail of the file."""

    schemas: Mapping[int, SchemaRecord]
    channels: Mapping[int, ChannelRecord]
    chunk_indexes: Sequence[ChunkIndexRecord]
    statistics: StatisticsRecord | None
    profile: str = ""
    library: str = ""
    bytes_read: int = 0
    range_requests: int = 0
    _source: ByteSource | None = field(default=None, repr=False, compare=False)

    @property
    def is_chunked(self) -> bool:
        """Whether the file has a usable chunk index.

        Unchunked or index-less files (truncated recordings, rosbag2's ``fastwrite`` preset) cannot
        be range-read for playback and must go through the ingest pipeline instead.
        """
        return len(self.chunk_indexes) > 0

    @property
    def message_start_time(self) -> int | None:
        if self.statistics is not None and self.statistics.message_count > 0:
            return self.statistics.message_start_time
        if self.chunk_indexes:
            return min(c.message_start_time for c in self.chunk_indexes)
        return None

    @property
    def message_end_time(self) -> int | None:
        if self.statistics is not None and self.statistics.message_count > 0:
            return self.statistics.message_end_time
        if self.chunk_indexes:
            return max(c.message_end_time for c in self.chunk_indexes)
        return None

    def channel_time_bounds(self, channel_id: int) -> tuple[int, int] | None:
        """Time bounds for one channel, narrowed to the chunks that actually contain it."""
        relevant = [c for c in self.chunk_indexes if channel_id in c.message_index_offsets]
        if not relevant:
            start, end = self.message_start_time, self.message_end_time
            return None if start is None or end is None else (start, end)
        return (
            min(c.message_start_time for c in relevant),
            max(c.message_end_time for c in relevant),
        )

    def chunks_for_channel(self, channel_id: int) -> Sequence[ChunkIndexRecord]:
        """Chunks whose message index mentions ``channel_id``, in file order."""
        relevant = [c for c in self.chunk_indexes if channel_id in c.message_index_offsets]
        return sorted(relevant or list(self.chunk_indexes), key=lambda c: c.chunk_start_offset)

    def message_count(self, channel_id: int) -> int | None:
        if self.statistics is None:
            return None
        return self.statistics.channel_message_counts.get(channel_id)


def read_summary(source: ByteSource, *, read_header: bool = True) -> McapSummary:
    """Read the summary section of the MCAP behind ``source``.

    Args:
        source: Random-access view over the MCAP bytes.
        read_header: Also read the file header record for the profile and writer library. Costs one
            extra small ranged read.

    Raises:
        McapParseError: If the bytes are not a well-formed MCAP, or the summary section is absent.
    """
    if source.size < TAIL_SIZE + len(MAGIC):
        raise McapParseError(f"{source.uri} is too small ({source.size} bytes) to be an MCAP file")

    tail = source.read_range(source.size - TAIL_SIZE, TAIL_SIZE)
    if not tail.endswith(MAGIC):
        raise McapParseError(f"{source.uri} does not end with MCAP magic bytes; the file may be truncated")

    footer_records = iter_records(tail[: -len(MAGIC)])
    if not footer_records or footer_records[0][0] != Opcode.FOOTER:
        raise McapParseError(f"{source.uri} has no readable footer record")
    summary_start, _summary_offset_start, _crc = parse_footer(footer_records[0][1])

    if summary_start == 0:
        raise McapParseError(
            f"{source.uri} has no summary section. Writers may legally omit it (for example rosbag2's "
            "'fastwrite' preset, or a recording that was cut short); such files cannot be range-read "
            "and must be ingested instead."
        )

    summary_end = source.size - TAIL_SIZE
    summary_length = summary_end - summary_start
    if summary_length <= 0:
        raise McapParseError(f"{source.uri} reports an out-of-range summary section at offset {summary_start}")
    if summary_length > MAX_SUMMARY_BYTES:
        raise McapParseError(
            f"{source.uri} reports a {summary_length} byte summary section, above the {MAX_SUMMARY_BYTES} byte limit"
        )

    summary_bytes = source.read_range(summary_start, summary_length)

    schemas: dict[int, SchemaRecord] = {}
    channels: dict[int, ChannelRecord] = {}
    chunk_indexes: list[ChunkIndexRecord] = []
    statistics: StatisticsRecord | None = None

    for opcode, content in iter_records(summary_bytes):
        if opcode == Opcode.SCHEMA:
            schema = parse_schema(content)
            schemas[schema.id] = schema
        elif opcode == Opcode.CHANNEL:
            channel = parse_channel(content)
            channels[channel.id] = channel
        elif opcode == Opcode.CHUNK_INDEX:
            chunk_indexes.append(parse_chunk_index(content))
        elif opcode == Opcode.STATISTICS:
            statistics = parse_statistics(content)

    profile, library = ("", "")
    if read_header:
        profile, library = _read_header(source)

    chunk_indexes.sort(key=lambda c: (c.message_start_time, c.chunk_start_offset))
    return McapSummary(
        schemas=schemas,
        channels=channels,
        chunk_indexes=chunk_indexes,
        statistics=statistics,
        profile=profile,
        library=library,
        bytes_read=getattr(source, "bytes_read", 0),
        range_requests=getattr(source, "range_requests", 0),
        _source=source,
    )


def _read_header(source: ByteSource) -> tuple[str, str]:
    """Read the header record that follows the leading magic bytes."""
    from nominal.core._mcap.records import _Cursor

    head = source.read_range(0, min(4096, source.size))
    if not head.startswith(MAGIC):
        raise McapParseError(f"{source.uri} does not start with MCAP magic bytes")
    for opcode, content in iter_records(head[len(MAGIC) :]):
        if opcode == Opcode.HEADER:
            cursor = _Cursor(content)
            return cursor.string(), cursor.string()
        break
    return "", ""


def read_chunk(source: ByteSource, index: ChunkIndexRecord) -> ChunkRecord:
    """Fetch and parse one chunk record named by the chunk index."""
    raw = source.read_range(index.chunk_start_offset, index.chunk_length)
    records = iter_records(raw)
    if not records or records[0][0] != Opcode.CHUNK:
        raise McapParseError(f"expected a chunk record at offset {index.chunk_start_offset} in {source.uri}")
    return parse_chunk(records[0][1])


def decompress_chunk(chunk: ChunkRecord) -> bytes:
    """Decompress a chunk's inner records.

    ``zstd`` and ``lz4`` decompressors are imported lazily: they are only needed for the deeper
    codec/keyframe probe, never for channel discovery.

    Raises:
        McapSourceError: If the chunk uses a compression scheme whose decompressor is not installed.
    """
    compression = chunk.compression
    if compression == "":
        return chunk.records
    if compression == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise McapSourceError(
                "this MCAP uses zstd chunk compression; install zstandard (pip install 'nominal[mcap]') "
                "to probe its video topics"
            ) from exc
        return bytes(zstandard.ZstdDecompressor().decompress(chunk.records, max_output_size=chunk.uncompressed_size))
    if compression == "lz4":
        try:
            import lz4.frame
        except ImportError as exc:
            raise McapSourceError(
                "this MCAP uses lz4 chunk compression; install lz4 (pip install 'nominal[mcap]') "
                "to probe its video topics"
            ) from exc
        return bytes(lz4.frame.decompress(chunk.records))
    raise McapSourceError(f"unsupported MCAP chunk compression {compression!r}")


def iter_chunk_messages(chunk_body: bytes, channel_id: int | None = None) -> "list[MessageRecord]":
    """Parse the message records inside a decompressed chunk, optionally filtered to one channel."""
    messages: list[MessageRecord] = []
    for opcode, content in iter_records(chunk_body):
        if opcode != Opcode.MESSAGE:
            continue
        message = parse_message(content)
        if channel_id is None or message.channel_id == channel_id:
            messages.append(message)
    return messages
