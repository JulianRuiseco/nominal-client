"""H.264/H.265 bitstream inspection, enough to answer "can a browser seek this?".

Direct playback hands compressed samples straight to the browser's ``VideoDecoder``, so whether a
file is playable is decided by the bitstream, not by our pipeline. Three properties matter:

* **Container shape** - Annex B start codes or AVCC length prefixes. WebCodecs needs to be told
  which, and files written before late 2023 by some recorders are length-prefixed.
* **Parameter sets** - a decoder cannot start without SPS/PPS. Some encoders emit them only once at
  stream start, which strands a player that seeks into the middle of a recording.
* **Keyframe cadence** - a seek must decode forward from the previous IDR. Sparse IDRs mean long
  seeks no client-side trick can fix. Counting I-frames overcounts: NVIDIA Jetson encoders emit
  non-seekable I-frames between sparse IDRs, so only true IDRs are counted here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterator, List, Sequence, Tuple


class VideoCodec(str, enum.Enum):
    """Codec named by the ``format`` field of a ``CompressedVideo`` message."""

    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str) -> "VideoCodec":
        normalized = raw.strip().lower().replace("-", "").replace(".", "").replace("_", "")
        if normalized in ("h264", "avc", "avc1", "x264", "264"):
            return cls.H264
        if normalized in ("h265", "hevc", "hvc1", "hev1", "x265", "265"):
            return cls.H265
        if normalized in ("vp9", "vp09"):
            return cls.VP9
        if normalized in ("av1", "av01"):
            return cls.AV1
        return cls.UNKNOWN


class BitstreamFormat(str, enum.Enum):
    """How NAL units are delimited inside a sample."""

    ANNEX_B = "annexb"
    LENGTH_PREFIXED = "length_prefixed"
    UNKNOWN = "unknown"


class FrameKind(str, enum.Enum):
    """Seek-relevant classification of one access unit."""

    KEYFRAME = "keyframe"
    """A true IDR: a decoder can start here with no prior frames."""

    INTRA_NON_IDR = "intra_non_idr"
    """An I-frame that is not an IDR. Looks like a keyframe to naive tooling, but is not seekable."""

    DELTA = "delta"
    UNKNOWN = "unknown"


# H.264 NAL unit types (ITU-T H.264 table 7-1).
_H264_NAL_SLICE_NON_IDR = 1
_H264_NAL_SLICE_PARTITION_A = 2
_H264_NAL_SLICE_IDR = 5
_H264_NAL_SEI = 6
_H264_NAL_SPS = 7
_H264_NAL_PPS = 8

# H.265 NAL unit types (ITU-T H.265 table 7-1).
_H265_IRAP_RANGE = range(16, 24)  # BLA_W_LP .. RSV_IRAP_VCL23
_H265_IDR_RANGE = range(19, 22)  # IDR_W_RADL, IDR_N_LP, CRA_NUT
_H265_NAL_VPS = 32
_H265_NAL_SPS = 33
_H265_NAL_PPS = 34

# slice_type values that indicate bidirectional prediction (ITU-T H.264 table 7-6).
_H264_B_SLICE_TYPES = (1, 6)
_H264_I_SLICE_TYPES = (2, 7)

_HIGH_PROFILE_IDCS = frozenset({100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135})
_SUB_WIDTH_C = {0: 1, 1: 2, 2: 2, 3: 1}
_SUB_HEIGHT_C = {0: 1, 1: 2, 2: 1, 3: 1}

MAX_NAL_SCAN_BYTES = 8 * 1024 * 1024
"""Cap on how much of a single sample is scanned, so a corrupt length prefix cannot spin."""


@dataclass(frozen=True)
class H264Sps:
    """The parts of an H.264 sequence parameter set that playback cares about."""

    profile_idc: int
    constraint_flags: int
    level_idc: int
    chroma_format_idc: int
    width: int
    height: int
    frame_mbs_only_flag: int
    max_num_ref_frames: int

    @property
    def codec_string(self) -> str:
        """The ``avc1.PPCCLL`` string WebCodecs wants in ``VideoDecoderConfig.codec``."""
        return f"avc1.{self.profile_idc:02x}{self.constraint_flags:02x}{self.level_idc:02x}"


@dataclass
class AccessUnitScan:
    """What one sample (one ``CompressedVideo`` message) contains."""

    format: BitstreamFormat = BitstreamFormat.UNKNOWN
    kind: FrameKind = FrameKind.UNKNOWN
    has_sps: bool = False
    has_pps: bool = False
    has_b_slices: bool = False
    nal_types: List[int] = field(default_factory=list)
    sps: H264Sps | None = None
    sps_bytes: bytes | None = field(default=None, repr=False)
    pps_bytes: bytes | None = field(default=None, repr=False)


class _BitReader:
    """RBSP bit reader with Exp-Golomb support."""

    __slots__ = ("_data", "_bit")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._bit = 0

    def _bits_left(self) -> int:
        return len(self._data) * 8 - self._bit

    def u(self, n: int) -> int:
        if self._bits_left() < n:
            raise ValueError("bitstream truncated")
        value = 0
        for _ in range(n):
            byte = self._data[self._bit >> 3]
            value = (value << 1) | ((byte >> (7 - (self._bit & 7))) & 1)
            self._bit += 1
        return value

    def ue(self) -> int:
        leading = 0
        while True:
            if self._bits_left() <= 0:
                raise ValueError("bitstream truncated reading ue(v)")
            if self.u(1):
                break
            leading += 1
            if leading > 32:
                raise ValueError("implausible Exp-Golomb prefix")
        return (1 << leading) - 1 + (self.u(leading) if leading else 0)

    def se(self) -> int:
        value = self.ue()
        return (value + 1) // 2 if value % 2 else -(value // 2)


def strip_emulation_prevention(data: bytes) -> bytes:
    """Remove ``0x03`` emulation-prevention bytes to recover the raw RBSP."""
    out = bytearray()
    zeros = 0
    for byte in data:
        if zeros >= 2 and byte == 0x03:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0x00 else 0
    return bytes(out)


def detect_format(sample: bytes) -> BitstreamFormat:
    """Guess whether a sample uses Annex B start codes or 4-byte length prefixes."""
    if len(sample) < 5:
        return BitstreamFormat.UNKNOWN
    if sample.startswith(b"\x00\x00\x01") or sample.startswith(b"\x00\x00\x00\x01"):
        # An AVCC sample whose first NAL is >= 16MB would also start with 0x00000001, which is not
        # a realistic frame size, so treating this as Annex B is safe.
        return BitstreamFormat.ANNEX_B
    length = int.from_bytes(sample[:4], "big")
    if 0 < length <= len(sample) - 4:
        return BitstreamFormat.LENGTH_PREFIXED
    return BitstreamFormat.UNKNOWN


def iter_nal_units(sample: bytes, fmt: BitstreamFormat) -> Iterator[bytes]:
    """Yield NAL units from a sample, without start codes or length prefixes."""
    if fmt == BitstreamFormat.LENGTH_PREFIXED:
        yield from _iter_length_prefixed(sample)
    elif fmt == BitstreamFormat.ANNEX_B:
        yield from _iter_annex_b(sample)


def _iter_length_prefixed(sample: bytes) -> Iterator[bytes]:
    pos = 0
    total = min(len(sample), MAX_NAL_SCAN_BYTES)
    while pos + 4 <= total:
        length = int.from_bytes(sample[pos : pos + 4], "big")
        pos += 4
        if length <= 0 or pos + length > len(sample):
            return
        yield sample[pos : pos + length]
        pos += length


def _iter_annex_b(sample: bytes) -> Iterator[bytes]:
    starts: List[Tuple[int, int]] = []
    pos = 0
    total = min(len(sample), MAX_NAL_SCAN_BYTES)
    while pos < total:
        idx = sample.find(b"\x00\x00\x01", pos)
        if idx < 0:
            break
        prefix = 4 if idx > 0 and sample[idx - 1] == 0x00 else 3
        starts.append((idx + 3, prefix))
        pos = idx + 3
    for i, (payload_start, _prefix) in enumerate(starts):
        end = len(sample) if i + 1 == len(starts) else starts[i + 1][0] - 3
        # Trim the trailing zero byte of a 4-byte start code belonging to the next NAL.
        while end > payload_start and sample[end - 1] == 0x00:
            end -= 1
        if end > payload_start:
            yield sample[payload_start:end]


def parse_h264_sps(nal: bytes) -> H264Sps:
    """Parse an H.264 SPS NAL (without start code) into its playback-relevant fields.

    Raises:
        ValueError: If the NAL is not an SPS or is truncated.
    """
    if not nal:
        raise ValueError("empty SPS NAL")
    if nal[0] & 0x1F != _H264_NAL_SPS:
        raise ValueError(f"NAL type {nal[0] & 0x1F} is not an SPS")

    reader = _BitReader(strip_emulation_prevention(nal[1:]))
    profile_idc = reader.u(8)
    constraint_flags = reader.u(8)
    level_idc = reader.u(8)
    reader.ue()  # seq_parameter_set_id

    chroma_format_idc = _read_chroma_prelude(reader, profile_idc)

    reader.ue()  # log2_max_frame_num_minus4
    _skip_pic_order_cnt(reader)

    max_num_ref_frames = reader.ue()
    reader.u(1)  # gaps_in_frame_num_value_allowed_flag
    pic_width_in_mbs_minus1 = reader.ue()
    pic_height_in_map_units_minus1 = reader.ue()
    frame_mbs_only_flag = reader.u(1)
    if not frame_mbs_only_flag:
        reader.u(1)  # mb_adaptive_frame_field_flag
    reader.u(1)  # direct_8x8_inference_flag

    width, height = _read_frame_size(
        reader,
        pic_width_in_mbs_minus1=pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1=pic_height_in_map_units_minus1,
        frame_mbs_only_flag=frame_mbs_only_flag,
        chroma_format_idc=chroma_format_idc,
    )

    return H264Sps(
        profile_idc=profile_idc,
        constraint_flags=constraint_flags,
        level_idc=level_idc,
        chroma_format_idc=chroma_format_idc,
        width=max(width, 0),
        height=max(height, 0),
        frame_mbs_only_flag=frame_mbs_only_flag,
        max_num_ref_frames=max_num_ref_frames,
    )


def _read_chroma_prelude(reader: _BitReader, profile_idc: int) -> int:
    """Read the fields that only high profiles carry, returning ``chroma_format_idc``."""
    if profile_idc not in _HIGH_PROFILE_IDCS:
        return 1  # 4:2:0 is implied for baseline/main/extended.
    chroma_format_idc = reader.ue()
    if chroma_format_idc == 3:
        reader.u(1)  # separate_colour_plane_flag
    reader.ue()  # bit_depth_luma_minus8
    reader.ue()  # bit_depth_chroma_minus8
    reader.u(1)  # qpprime_y_zero_transform_bypass_flag
    if reader.u(1):  # seq_scaling_matrix_present_flag
        for i in range(8 if chroma_format_idc != 3 else 12):
            if reader.u(1):
                _skip_scaling_list(reader, 16 if i < 6 else 64)
    return chroma_format_idc


def _skip_pic_order_cnt(reader: _BitReader) -> None:
    """Consume the picture-order-count fields, whose shape depends on the count type."""
    pic_order_cnt_type = reader.ue()
    if pic_order_cnt_type == 0:
        reader.ue()  # log2_max_pic_order_cnt_lsb_minus4
    elif pic_order_cnt_type == 1:
        reader.u(1)  # delta_pic_order_always_zero_flag
        reader.se()  # offset_for_non_ref_pic
        reader.se()  # offset_for_top_to_bottom_field
        for _ in range(reader.ue()):
            reader.se()


def _read_frame_size(
    reader: _BitReader,
    *,
    pic_width_in_mbs_minus1: int,
    pic_height_in_map_units_minus1: int,
    frame_mbs_only_flag: int,
    chroma_format_idc: int,
) -> Tuple[int, int]:
    """Apply the cropping window to the macroblock grid to get the displayed frame size."""
    crop_left = crop_right = crop_top = crop_bottom = 0
    if reader.u(1):  # frame_cropping_flag
        crop_left = reader.ue()
        crop_right = reader.ue()
        crop_top = reader.ue()
        crop_bottom = reader.ue()

    width = (pic_width_in_mbs_minus1 + 1) * 16
    height = (2 - frame_mbs_only_flag) * (pic_height_in_map_units_minus1 + 1) * 16
    if chroma_format_idc == 0:
        crop_unit_x, crop_unit_y = 1, 2 - frame_mbs_only_flag
    else:
        crop_unit_x = _SUB_WIDTH_C[chroma_format_idc]
        crop_unit_y = _SUB_HEIGHT_C[chroma_format_idc] * (2 - frame_mbs_only_flag)
    return (
        width - crop_unit_x * (crop_left + crop_right),
        height - crop_unit_y * (crop_top + crop_bottom),
    )


def _skip_scaling_list(reader: _BitReader, size: int) -> None:
    last_scale = 8
    next_scale = 8
    for _ in range(size):
        if next_scale != 0:
            next_scale = (last_scale + reader.se() + 256) % 256
        last_scale = next_scale if next_scale != 0 else last_scale


def scan_access_unit(sample: bytes, codec: VideoCodec, fmt: BitstreamFormat | None = None) -> AccessUnitScan:
    """Classify a single compressed sample.

    Args:
        sample: The ``data`` payload of one ``CompressedVideo`` message.
        codec: Codec declared by the message's ``format`` field.
        fmt: Bitstream shape, if already known. Detected from the sample when omitted.
    """
    resolved = fmt if fmt is not None and fmt != BitstreamFormat.UNKNOWN else detect_format(sample)
    scan = AccessUnitScan(format=resolved)
    if resolved == BitstreamFormat.UNKNOWN:
        return scan

    nals = list(iter_nal_units(sample, resolved))
    if codec == VideoCodec.H265:
        return _scan_h265(scan, nals)
    if codec == VideoCodec.H264:
        return _scan_h264(scan, nals)
    scan.nal_types = [nal[0] for nal in nals if nal]
    return scan


def _scan_h264(scan: AccessUnitScan, nals: Sequence[bytes]) -> AccessUnitScan:
    saw_idr = saw_intra_slice = saw_slice = False
    for nal in nals:
        if not nal:
            continue
        nal_type = nal[0] & 0x1F
        scan.nal_types.append(nal_type)
        if nal_type in (_H264_NAL_SPS, _H264_NAL_PPS):
            _record_h264_parameter_set(scan, nal, nal_type)
        elif nal_type == _H264_NAL_SLICE_IDR:
            saw_idr = saw_slice = True
        elif nal_type in (_H264_NAL_SLICE_NON_IDR, _H264_NAL_SLICE_PARTITION_A):
            saw_slice = True
            slice_type = _peek_slice_type(nal)
            if slice_type in _H264_B_SLICE_TYPES:
                scan.has_b_slices = True
            elif slice_type in _H264_I_SLICE_TYPES:
                saw_intra_slice = True

    scan.kind = _frame_kind(saw_idr=saw_idr, saw_intra=saw_intra_slice, saw_slice=saw_slice)
    return scan


def _record_h264_parameter_set(scan: AccessUnitScan, nal: bytes, nal_type: int) -> None:
    if nal_type == _H264_NAL_SPS:
        scan.has_sps = True
        scan.sps_bytes = nal
        try:
            scan.sps = parse_h264_sps(nal)
        except ValueError:
            scan.sps = None
    else:
        scan.has_pps = True
        scan.pps_bytes = nal


def _frame_kind(*, saw_idr: bool, saw_intra: bool, saw_slice: bool) -> FrameKind:
    if saw_idr:
        return FrameKind.KEYFRAME
    if saw_intra:
        return FrameKind.INTRA_NON_IDR
    if saw_slice:
        return FrameKind.DELTA
    return FrameKind.UNKNOWN


def _peek_slice_type(nal: bytes) -> int | None:
    """Read ``slice_type`` from the front of a slice header."""
    try:
        reader = _BitReader(strip_emulation_prevention(nal[1:]))
        reader.ue()  # first_mb_in_slice
        return reader.ue()
    except (ValueError, IndexError):
        return None


def _scan_h265(scan: AccessUnitScan, nals: Sequence[bytes]) -> AccessUnitScan:
    saw_idr = saw_irap = saw_slice = False
    for nal in nals:
        if len(nal) < 2:
            continue
        # H.265 uses a two-byte NAL header; the type is bits 1-6 of the first byte.
        nal_type = (nal[0] >> 1) & 0x3F
        scan.nal_types.append(nal_type)
        if nal_type == _H265_NAL_SPS:
            scan.has_sps = True
            scan.sps_bytes = nal
        elif nal_type == _H265_NAL_PPS:
            scan.has_pps = True
            scan.pps_bytes = nal
        elif nal_type < 32:
            saw_slice = True
            saw_idr = saw_idr or nal_type in _H265_IDR_RANGE
            saw_irap = saw_irap or nal_type in _H265_IRAP_RANGE

    scan.kind = _frame_kind(saw_idr=saw_idr, saw_intra=saw_irap, saw_slice=saw_slice)
    return scan
