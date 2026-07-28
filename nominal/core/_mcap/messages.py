"""Decode ``CompressedVideo`` messages from both wire encodings, without a protobuf dependency.

The same video bytes reach us in two envelopes:

* protobuf ``foxglove.CompressedVideo`` - what the Foxglove SDK and most custom recorders write.
* CDR-encoded ``foxglove_msgs/msg/CompressedVideo`` - what ROS 2's rosbag2 always writes.

Only four fields matter (timestamp, frame_id, data, format), so both are parsed directly off the
wire rather than through generated message classes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Tuple


class MessageDecodeError(ValueError):
    """Raised when a message body does not decode as a ``CompressedVideo``."""


@dataclass(frozen=True)
class CompressedVideoMessage:
    """The decoded contents of one ``CompressedVideo`` message."""

    timestamp_ns: int | None
    frame_id: str
    format: str
    data: bytes = field(repr=False)


# ---------------------------------------------------------------------------
# protobuf


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            break
    raise MessageDecodeError("truncated or malformed protobuf varint")


def _skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = _read_varint(buf, pos)
        return pos
    if wire_type == 1:
        return pos + 8
    if wire_type == 2:
        length, pos = _read_varint(buf, pos)
        return pos + length
    if wire_type == 5:
        return pos + 4
    raise MessageDecodeError(f"unsupported protobuf wire type {wire_type}")


def decode_protobuf_compressed_video(buf: bytes) -> CompressedVideoMessage:
    """Decode a protobuf-encoded ``foxglove.CompressedVideo``."""
    timestamp_ns: int | None = None
    frame_id = ""
    fmt = ""
    data = b""

    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field_number, wire_type = key >> 3, key & 0x07
        if wire_type == 2:
            length, pos = _read_varint(buf, pos)
            end = pos + length
            if end > len(buf):
                raise MessageDecodeError("truncated protobuf length-delimited field")
            payload = buf[pos:end]
            pos = end
            if field_number == 1:
                timestamp_ns = _decode_protobuf_timestamp(payload)
            elif field_number == 2:
                frame_id = payload.decode("utf-8", errors="replace")
            elif field_number == 3:
                data = payload
            elif field_number == 4:
                fmt = payload.decode("utf-8", errors="replace")
        else:
            pos = _skip_field(buf, pos, wire_type)

    if not data:
        raise MessageDecodeError("protobuf CompressedVideo carried no data field")
    return CompressedVideoMessage(timestamp_ns=timestamp_ns, frame_id=frame_id, format=fmt, data=data)


def _decode_protobuf_timestamp(buf: bytes) -> int | None:
    seconds = 0
    nanos = 0
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field_number, wire_type = key >> 3, key & 0x07
        if wire_type == 0:
            value, pos = _read_varint(buf, pos)
            if field_number == 1:
                seconds = _zigzag_to_signed(value)
            elif field_number == 2:
                nanos = _zigzag_to_signed(value)
        else:
            pos = _skip_field(buf, pos, wire_type)
    if seconds == 0 and nanos == 0:
        return None
    return seconds * 1_000_000_000 + nanos


def _zigzag_to_signed(value: int) -> int:
    """Interpret a protobuf varint as int64/int32 (two's complement, not zigzag)."""
    if value >= 1 << 63:
        return value - (1 << 64)
    return value


# ---------------------------------------------------------------------------
# CDR (ROS 2)


class _CdrReader:
    """Little/big-endian CDR reader with the alignment rules ROS 2 uses."""

    __slots__ = ("_buf", "_pos", "_endian")

    def __init__(self, buf: bytes) -> None:
        if len(buf) < 4:
            raise MessageDecodeError("CDR message shorter than its encapsulation header")
        # Encapsulation: 0x00 0x00 = CDR_BE, 0x00 0x01 = CDR_LE, then two option bytes.
        self._endian = "<" if buf[1] & 0x01 else ">"
        self._buf = buf
        self._pos = 4

    def _align(self, size: int) -> None:
        # Alignment is measured from the end of the 4-byte encapsulation header.
        offset = self._pos - 4
        padding = (size - (offset % size)) % size
        self._pos += padding

    def _take(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._buf):
            raise MessageDecodeError(f"truncated CDR message: wanted {n} bytes at {self._pos}")
        out = self._buf[self._pos : end]
        self._pos = end
        return out

    def int32(self) -> int:
        self._align(4)
        return int(struct.unpack(self._endian + "i", self._take(4))[0])

    def uint32(self) -> int:
        self._align(4)
        return int(struct.unpack(self._endian + "I", self._take(4))[0])

    def string(self) -> str:
        length = self.uint32()
        raw = self._take(length)
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def byte_sequence(self) -> bytes:
        return self._take(self.uint32())


def decode_cdr_compressed_video(buf: bytes) -> CompressedVideoMessage:
    """Decode a CDR-encoded ``foxglove_msgs/msg/CompressedVideo`` as written by rosbag2.

    Field order matches the message definition: ``timestamp``, ``frame_id``, ``data``, ``format``.
    """
    reader = _CdrReader(buf)
    seconds = reader.int32()
    nanoseconds = reader.uint32()
    frame_id = reader.string()
    data = reader.byte_sequence()
    fmt = reader.string()

    timestamp_ns: int | None = seconds * 1_000_000_000 + nanoseconds
    if seconds == 0 and nanoseconds == 0:
        timestamp_ns = None
    if not data:
        raise MessageDecodeError("CDR CompressedVideo carried no data field")
    return CompressedVideoMessage(timestamp_ns=timestamp_ns, frame_id=frame_id, format=fmt, data=data)
