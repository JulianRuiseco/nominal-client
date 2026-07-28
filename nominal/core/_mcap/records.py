"""Record-level primitives for the MCAP binary format.

Reference: https://mcap.dev/spec

Every record is ``opcode:uint8`` followed by ``length:uint64`` and that many bytes of content.
Strings and byte arrays are length-prefixed; maps and arrays are byte-length prefixed.
All integers are little-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Mapping, Tuple

MAGIC = b"\x89MCAP0\r\n"
"""Magic bytes at both the start and the end of a well-formed MCAP file."""

FOOTER_RECORD_SIZE = 1 + 8 + 20
"""opcode + length prefix + (summaryStart, summaryOffsetStart, summaryCrc)."""

TAIL_SIZE = FOOTER_RECORD_SIZE + len(MAGIC)
"""Bytes at the end of the file that hold the footer record and trailing magic."""


class Opcode:
    """MCAP record opcodes."""

    HEADER = 0x01
    FOOTER = 0x02
    SCHEMA = 0x03
    CHANNEL = 0x04
    MESSAGE = 0x05
    CHUNK = 0x06
    MESSAGE_INDEX = 0x07
    CHUNK_INDEX = 0x08
    ATTACHMENT = 0x09
    ATTACHMENT_INDEX = 0x0A
    STATISTICS = 0x0B
    METADATA = 0x0C
    METADATA_INDEX = 0x0D
    SUMMARY_OFFSET = 0x0E
    DATA_END = 0x0F


class McapParseError(ValueError):
    """Raised when a byte range does not parse as valid MCAP."""


@dataclass(frozen=True)
class SchemaRecord:
    id: int
    name: str
    encoding: str
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class ChannelRecord:
    id: int
    schema_id: int
    topic: str
    message_encoding: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class ChunkIndexRecord:
    message_start_time: int
    message_end_time: int
    chunk_start_offset: int
    chunk_length: int
    message_index_offsets: Mapping[int, int]
    message_index_length: int
    compression: str
    compressed_size: int
    uncompressed_size: int


@dataclass(frozen=True)
class StatisticsRecord:
    message_count: int
    schema_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    chunk_count: int
    message_start_time: int
    message_end_time: int
    channel_message_counts: Mapping[int, int]


@dataclass(frozen=True)
class MessageRecord:
    channel_id: int
    sequence: int
    log_time: int
    publish_time: int
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class ChunkRecord:
    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    compression: str
    records: bytes = field(repr=False)


class _Cursor:
    """A bounds-checked little-endian reader over a byte buffer."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self._buf = buf
        self._pos = pos

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def _take(self, n: int) -> bytes:
        if n < 0:
            raise McapParseError(f"negative read length {n}")
        end = self._pos + n
        if end > len(self._buf):
            raise McapParseError(f"truncated record: wanted {n} bytes at offset {self._pos}, have {self.remaining}")
        chunk = self._buf[self._pos : end]
        self._pos = end
        return chunk

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return int(struct.unpack_from("<H", self._take(2))[0])

    def u32(self) -> int:
        return int(struct.unpack_from("<I", self._take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack_from("<Q", self._take(8))[0])

    def string(self) -> str:
        raw = self._take(self.u32())
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise McapParseError(f"invalid utf-8 in MCAP string: {exc}") from exc

    def byte_array32(self) -> bytes:
        return self._take(self.u32())

    def byte_array64(self) -> bytes:
        return self._take(self.u64())

    def rest(self) -> bytes:
        return self._take(self.remaining)

    def string_map(self) -> Mapping[str, str]:
        body = _Cursor(self._take(self.u32()))
        out: dict[str, str] = {}
        while body.remaining:
            key = body.string()
            out[key] = body.string()
        return out

    def u16_u64_map(self) -> Mapping[int, int]:
        body = _Cursor(self._take(self.u32()))
        out: dict[int, int] = {}
        while body.remaining:
            key = body.u16()
            out[key] = body.u64()
        return out


def iter_records(buf: bytes) -> "list[Tuple[int, bytes]]":
    """Split a buffer of concatenated MCAP records into ``(opcode, content)`` pairs.

    A trailing partial record is dropped rather than raising, so this is safe to use on a byte
    range that was cut mid-record (which is what a ranged read of a chunk gives you).
    """
    out: list[Tuple[int, bytes]] = []
    pos = 0
    total = len(buf)
    while pos + 9 <= total:
        opcode = buf[pos]
        (length,) = struct.unpack_from("<Q", buf, pos + 1)
        start = pos + 9
        end = start + length
        if end > total:
            break
        out.append((opcode, buf[start:end]))
        pos = end
    return out


def parse_schema(content: bytes) -> SchemaRecord:
    c = _Cursor(content)
    return SchemaRecord(id=c.u16(), name=c.string(), encoding=c.string(), data=c.byte_array32())


def parse_channel(content: bytes) -> ChannelRecord:
    c = _Cursor(content)
    return ChannelRecord(
        id=c.u16(),
        schema_id=c.u16(),
        topic=c.string(),
        message_encoding=c.string(),
        metadata=c.string_map(),
    )


def parse_chunk_index(content: bytes) -> ChunkIndexRecord:
    c = _Cursor(content)
    return ChunkIndexRecord(
        message_start_time=c.u64(),
        message_end_time=c.u64(),
        chunk_start_offset=c.u64(),
        chunk_length=c.u64(),
        message_index_offsets=c.u16_u64_map(),
        message_index_length=c.u64(),
        compression=c.string(),
        compressed_size=c.u64(),
        uncompressed_size=c.u64(),
    )


def parse_statistics(content: bytes) -> StatisticsRecord:
    c = _Cursor(content)
    return StatisticsRecord(
        message_count=c.u64(),
        schema_count=c.u16(),
        channel_count=c.u32(),
        attachment_count=c.u32(),
        metadata_count=c.u32(),
        chunk_count=c.u32(),
        message_start_time=c.u64(),
        message_end_time=c.u64(),
        channel_message_counts=c.u16_u64_map(),
    )


def parse_message(content: bytes) -> MessageRecord:
    c = _Cursor(content)
    return MessageRecord(
        channel_id=c.u16(),
        sequence=c.u32(),
        log_time=c.u64(),
        publish_time=c.u64(),
        data=c.rest(),
    )


def parse_chunk(content: bytes) -> ChunkRecord:
    c = _Cursor(content)
    return ChunkRecord(
        message_start_time=c.u64(),
        message_end_time=c.u64(),
        uncompressed_size=c.u64(),
        uncompressed_crc=c.u32(),
        compression=c.string(),
        records=c.byte_array64(),
    )


def parse_footer(content: bytes) -> Tuple[int, int, int]:
    """Parse a footer record into ``(summary_start, summary_offset_start, summary_crc)``."""
    c = _Cursor(content)
    return c.u64(), c.u64(), c.u32()
