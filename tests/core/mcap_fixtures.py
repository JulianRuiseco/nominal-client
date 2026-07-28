"""Builders for synthetic MCAP files used by the direct-video tests.

Writing MCAPs by hand (rather than pulling in a writer library) keeps the tests honest about the
byte layout the reader claims to understand.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from nominal.core._mcap.records import MAGIC, Opcode

PROTOBUF_SCHEMA = "foxglove.CompressedVideo"
ROS_SCHEMA = "foxglove_msgs/msg/CompressedVideo"


def _record(opcode: int, content: bytes) -> bytes:
    return bytes([opcode]) + struct.pack("<Q", len(content)) + content


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _string_map(values: dict[str, str]) -> bytes:
    body = b"".join(_string(k) + _string(v) for k, v in values.items())
    return struct.pack("<I", len(body)) + body


def _u16_u64_map(values: dict[int, int]) -> bytes:
    body = b"".join(struct.pack("<HQ", k, v) for k, v in values.items())
    return struct.pack("<I", len(body)) + body


# ---------------------------------------------------------------------------
# message encoders


def encode_protobuf_compressed_video(data: bytes, fmt: str, timestamp_ns: int | None = None) -> bytes:
    out = bytearray()
    if timestamp_ns is not None:
        seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
        ts = bytearray()
        ts += b"\x08" + _varint(seconds)
        ts += b"\x10" + _varint(nanos)
        out += b"\x0a" + _varint(len(ts)) + bytes(ts)
    out += b"\x12" + _varint(0)  # frame_id: ""
    out += b"\x1a" + _varint(len(data)) + data
    raw_format = fmt.encode()
    out += b"\x22" + _varint(len(raw_format)) + raw_format
    return bytes(out)


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_cdr_compressed_video(data: bytes, fmt: str, timestamp_ns: int | None = None) -> bytes:
    seconds, nanos = divmod(timestamp_ns or 0, 1_000_000_000)
    body = bytearray()

    def align(size: int) -> None:
        padding = (size - (len(body) % size)) % size
        body.extend(b"\x00" * padding)

    align(4)
    body += struct.pack("<i", seconds)
    align(4)
    body += struct.pack("<I", nanos)
    # frame_id: length-prefixed, null terminated
    align(4)
    body += struct.pack("<I", 1) + b"\x00"
    align(4)
    body += struct.pack("<I", len(data)) + data
    raw_format = fmt.encode() + b"\x00"
    align(4)
    body += struct.pack("<I", len(raw_format)) + raw_format
    return b"\x00\x01\x00\x00" + bytes(body)


# ---------------------------------------------------------------------------
# H.264 sample builders
#
# NAL units here are generated bit by bit rather than pasted as hex, so a test that reads
# 1920x1080 back out of an SPS is genuinely exercising the parser.


class BitWriter:
    """Writes fixed-width and Exp-Golomb fields, MSB first."""

    def __init__(self) -> None:
        """Start with an empty bit buffer."""
        self._bits: List[int] = []

    def u(self, count: int, value: int) -> "BitWriter":
        for shift in range(count - 1, -1, -1):
            self._bits.append((value >> shift) & 1)
        return self

    def ue(self, value: int) -> "BitWriter":
        code = value + 1
        length = code.bit_length()
        self.u(length - 1, 0)
        self.u(length, code)
        return self

    def rbsp_trailing(self) -> "BitWriter":
        self._bits.append(1)
        while len(self._bits) % 8:
            self._bits.append(0)
        return self

    def bytes(self) -> bytes:
        out = bytearray()
        for i in range(0, len(self._bits), 8):
            byte = 0
            for bit in self._bits[i : i + 8]:
                byte = (byte << 1) | bit
            byte <<= 8 - len(self._bits[i : i + 8])
            out.append(byte)
        return bytes(out)


def add_emulation_prevention(rbsp: bytes) -> bytes:
    """Insert the ``0x03`` bytes an encoder would emit, so the parser has to strip them."""
    out = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 0x03:
            out.append(0x03)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0x00 else 0
    return bytes(out)


def make_sps(width: int = 1920, height: int = 1080, profile_idc: int = 66, level_idc: int = 31) -> bytes:
    """Build a valid baseline-profile SPS NAL for the given frame size."""
    mbs_wide = (width + 15) // 16
    mbs_high = (height + 15) // 16
    # 4:2:0 progressive: CropUnitX = 2, CropUnitY = 2.
    crop_right = (mbs_wide * 16 - width) // 2
    crop_bottom = (mbs_high * 16 - height) // 2

    writer = BitWriter()
    writer.u(8, profile_idc).u(8, 0xC0).u(8, level_idc)
    writer.ue(0)  # seq_parameter_set_id
    writer.ue(0)  # log2_max_frame_num_minus4
    writer.ue(2)  # pic_order_cnt_type = 2 (no extra fields)
    writer.ue(1)  # max_num_ref_frames
    writer.u(1, 0)  # gaps_in_frame_num_value_allowed_flag
    writer.ue(mbs_wide - 1)
    writer.ue(mbs_high - 1)
    writer.u(1, 1)  # frame_mbs_only_flag
    writer.u(1, 1)  # direct_8x8_inference_flag
    if crop_right or crop_bottom:
        writer.u(1, 1)  # frame_cropping_flag
        writer.ue(0).ue(crop_right).ue(0).ue(crop_bottom)
    else:
        writer.u(1, 0)
    writer.u(1, 0)  # vui_parameters_present_flag
    writer.rbsp_trailing()
    return bytes([0x67]) + add_emulation_prevention(writer.bytes())


def make_pps() -> bytes:
    writer = BitWriter()
    writer.ue(0)  # pic_parameter_set_id
    writer.ue(0)  # seq_parameter_set_id
    writer.u(1, 0)  # entropy_coding_mode_flag
    writer.u(1, 0)  # bottom_field_pic_order_in_frame_present_flag
    writer.ue(0)  # num_slice_groups_minus1
    writer.ue(0).ue(0)  # num_ref_idx_l0/l1_default_active_minus1
    writer.u(1, 0)  # weighted_pred_flag
    writer.u(2, 0)  # weighted_bipred_idc
    writer.ue(0).ue(0).ue(0)  # pic_init_qp/qs_minus26, chroma_qp_index_offset (se(v) of 0 == ue(0))
    writer.u(1, 1)  # deblocking_filter_control_present_flag
    writer.u(1, 0)  # constrained_intra_pred_flag
    writer.u(1, 0)  # redundant_pic_cnt_present_flag
    writer.rbsp_trailing()
    return bytes([0x68]) + add_emulation_prevention(writer.bytes())


def _slice_nal(nal_header: int, slice_type: int) -> bytes:
    writer = BitWriter()
    writer.ue(0)  # first_mb_in_slice
    writer.ue(slice_type)
    writer.ue(0)  # pic_parameter_set_id
    writer.u(4, 0)  # frame_num (log2_max_frame_num_minus4 = 0 -> 4 bits)
    writer.rbsp_trailing()
    return bytes([nal_header]) + add_emulation_prevention(writer.bytes())


SPS_1920x1080 = make_sps(1920, 1080)
PPS = make_pps()


def idr_slice() -> bytes:
    """NAL type 5 (IDR), slice_type 7 (I)."""
    return _slice_nal(0x65, 7)


def p_slice() -> bytes:
    """NAL type 1, slice_type 5 (P)."""
    return _slice_nal(0x41, 5)


def b_slice() -> bytes:
    """NAL type 1, slice_type 6 (B)."""
    return _slice_nal(0x41, 6)


def i_slice_non_idr() -> bytes:
    """NAL type 1 with slice_type 7 (I): intra coded, but not a seek point."""
    return _slice_nal(0x41, 7)


def annex_b(*nals: bytes) -> bytes:
    return b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


def length_prefixed(*nals: bytes) -> bytes:
    return b"".join(struct.pack(">I", len(nal)) + nal for nal in nals)


def keyframe_sample(*, with_parameter_sets: bool = True, prefixed: bool = False) -> bytes:
    nals: List[bytes] = []
    if with_parameter_sets:
        nals.extend([SPS_1920x1080, PPS])
    nals.append(idr_slice())
    return length_prefixed(*nals) if prefixed else annex_b(*nals)


def delta_sample(*, prefixed: bool = False, bidirectional: bool = False, intra: bool = False) -> bytes:
    if intra:
        nal = i_slice_non_idr()
    else:
        nal = b_slice() if bidirectional else p_slice()
    return length_prefixed(nal) if prefixed else annex_b(nal)


# ---------------------------------------------------------------------------
# MCAP builder


@dataclass
class McapBuilder:
    """Assemble a chunked, indexed MCAP with a summary section."""

    profile: str = "x-nominal-test"
    library: str = "nominal-test-writer"
    compression: str = ""
    _schemas: List[Tuple[int, str, str]] = field(default_factory=list)
    _channels: List[Tuple[int, int, str, str]] = field(default_factory=list)
    _messages: List[Tuple[int, int, bytes]] = field(default_factory=list)

    def add_schema(self, schema_id: int, name: str, encoding: str = "protobuf") -> "McapBuilder":
        self._schemas.append((schema_id, name, encoding))
        return self

    def add_channel(self, channel_id: int, schema_id: int, topic: str, message_encoding: str) -> "McapBuilder":
        self._channels.append((channel_id, schema_id, topic, message_encoding))
        return self

    def add_message(self, channel_id: int, log_time: int, data: bytes) -> "McapBuilder":
        self._messages.append((channel_id, log_time, data))
        return self

    def _schema_records(self) -> bytes:
        return b"".join(
            _record(Opcode.SCHEMA, struct.pack("<H", sid) + _string(name) + _string(enc) + struct.pack("<I", 0))
            for sid, name, enc in self._schemas
        )

    def _channel_records(self) -> bytes:
        return b"".join(
            _record(
                Opcode.CHANNEL,
                struct.pack("<HH", cid, sid) + _string(topic) + _string(enc) + _string_map({}),
            )
            for cid, sid, topic, enc in self._channels
        )

    def _compress(self, payload: bytes) -> bytes:
        if self.compression == "":
            return payload
        if self.compression == "zstd":
            import zstandard

            return bytes(zstandard.ZstdCompressor().compress(payload))
        if self.compression == "lz4":
            import lz4.frame

            return bytes(lz4.frame.compress(payload))
        raise ValueError(f"unsupported test compression {self.compression!r}")

    def build(self, *, messages_per_chunk: int = 1000, omit_summary: bool = False) -> bytes:
        out = bytearray(MAGIC)
        out += _record(Opcode.HEADER, _string(self.profile) + _string(self.library))

        schema_records = self._schema_records()
        channel_records = self._channel_records()
        out += schema_records
        out += channel_records

        ordered = sorted(self._messages, key=lambda m: m[1])
        chunk_indexes: List[bytes] = []
        for start in range(0, len(ordered), messages_per_chunk):
            group = ordered[start : start + messages_per_chunk]
            if not group:
                continue
            inner = bytearray()
            # (log_time, offset within the uncompressed chunk records) per channel, which is what a
            # MessageIndex record carries. Readers need these to seek inside a chunk; a chunk index
            # that points at nothing is rejected by spec-conformant readers.
            index_entries: dict[int, List[Tuple[int, int]]] = {}
            for channel_id, log_time, data in group:
                index_entries.setdefault(channel_id, []).append((log_time, len(inner)))
                inner += _record(
                    Opcode.MESSAGE,
                    struct.pack("<HIQQ", channel_id, 0, log_time, log_time) + data,
                )
            uncompressed = bytes(inner)
            compressed = self._compress(uncompressed)
            chunk_content = (
                struct.pack("<QQQI", group[0][1], group[-1][1], len(uncompressed), 0)
                + _string(self.compression)
                + struct.pack("<Q", len(compressed))
                + compressed
            )
            chunk_offset = len(out)
            chunk_record = _record(Opcode.CHUNK, chunk_content)
            out += chunk_record

            # MessageIndex records follow their chunk in the data section.
            message_index_offsets: dict[int, int] = {}
            message_index_length = 0
            for channel_id, entries in index_entries.items():
                body = b"".join(struct.pack("<QQ", log_time, offset) for log_time, offset in entries)
                record = _record(
                    Opcode.MESSAGE_INDEX,
                    struct.pack("<H", channel_id) + struct.pack("<I", len(body)) + body,
                )
                message_index_offsets[channel_id] = len(out)
                out += record
                message_index_length += len(record)

            chunk_indexes.append(
                _record(
                    Opcode.CHUNK_INDEX,
                    struct.pack("<QQQQ", group[0][1], group[-1][1], chunk_offset, len(chunk_record))
                    + _u16_u64_map(message_index_offsets)
                    + struct.pack("<Q", message_index_length)
                    + _string(self.compression)
                    + struct.pack("<QQ", len(compressed), len(uncompressed)),
                )
            )

        out += _record(Opcode.DATA_END, struct.pack("<I", 0))

        if omit_summary:
            out += _record(Opcode.FOOTER, struct.pack("<QQI", 0, 0, 0))
            out += MAGIC
            return bytes(out)

        summary_start = len(out)
        out += schema_records
        out += channel_records
        out += b"".join(chunk_indexes)
        counts: dict[int, int] = {}
        for channel_id, _log_time, _data in ordered:
            counts[channel_id] = counts.get(channel_id, 0) + 1
        out += _record(
            Opcode.STATISTICS,
            struct.pack(
                "<QHIIIIQQ",
                len(ordered),
                len(self._schemas),
                len(self._channels),
                0,
                0,
                len(chunk_indexes),
                ordered[0][1] if ordered else 0,
                ordered[-1][1] if ordered else 0,
            )
            + _u16_u64_map(counts),
        )
        out += _record(Opcode.FOOTER, struct.pack("<QQI", summary_start, 0, 0))
        out += MAGIC
        return bytes(out)


def build_video_mcap(
    *,
    topic: str = "/camera/front",
    encoding: str = "protobuf",
    codec: str = "h264",
    frame_count: int = 60,
    keyframe_every: int = 10,
    frame_interval_ns: int = 33_333_333,
    start_time: int = 1_700_000_000_000_000_000,
    parameter_sets_on_keyframes: bool = True,
    prefixed: bool = False,
    bidirectional: bool = False,
    compression: str = "",
    messages_per_chunk: int = 1000,
    extra_topics: Sequence[str] = (),
) -> bytes:
    """Build an MCAP containing one video topic with a controllable keyframe cadence."""
    is_ros = encoding == "cdr"
    builder = McapBuilder(compression=compression)
    builder.add_schema(1, ROS_SCHEMA if is_ros else PROTOBUF_SCHEMA, "ros2msg" if is_ros else "protobuf")
    builder.add_channel(1, 1, topic, "cdr" if is_ros else "protobuf")

    for i, extra in enumerate(extra_topics, start=2):
        builder.add_schema(i, "std_msgs/msg/Float64", "ros2msg")
        builder.add_channel(i, i, extra, "cdr")
        builder.add_message(i, start_time + i, b"\x00\x01\x00\x00\x00\x00\x00\x00")

    encode = encode_cdr_compressed_video if is_ros else encode_protobuf_compressed_video
    for frame in range(frame_count):
        log_time = start_time + frame * frame_interval_ns
        is_key = keyframe_every > 0 and frame % keyframe_every == 0
        if is_key:
            sample = keyframe_sample(with_parameter_sets=parameter_sets_on_keyframes or frame == 0, prefixed=prefixed)
        else:
            sample = delta_sample(prefixed=prefixed, bidirectional=bidirectional)
        builder.add_message(1, log_time, encode(sample, codec, log_time - 1_000_000))

    return builder.build(messages_per_chunk=messages_per_chunk)
