"""Pure-python MCAP inspection used to register video topics without copying or transcoding.

An MCAP file carries its own index: a summary section at the tail lists every channel, the file's
time bounds, and a chunk index mapping time ranges to byte ranges. Reading it takes a handful of
ranged reads, so a multi-gigabyte recording can be registered from kilobytes of traffic.

This package implements that read directly against the MCAP specification (https://mcap.dev/spec)
rather than depending on a third-party reader, so registration works with no extra dependencies.
Chunk decompression (needed only for the deeper codec/keyframe probe) uses ``zstandard``/``lz4``
when they are installed and degrades to an ``UNKNOWN`` classification when they are not.
"""

from nominal.core._mcap.bitstream import (
    BitstreamFormat,
    FrameKind,
    VideoCodec,
    parse_h264_sps,
    scan_access_unit,
)
from nominal.core._mcap.reader import McapSummary, read_summary
from nominal.core._mcap.records import (
    ChannelRecord,
    ChunkIndexRecord,
    McapParseError,
    SchemaRecord,
    StatisticsRecord,
)
from nominal.core._mcap.sources import (
    ByteSource,
    FileByteSource,
    HttpRangeByteSource,
    McapSourceError,
)
from nominal.core._mcap.video import (
    DirectPlaybackSupport,
    McapVideoChannel,
    McapVideoInspection,
    MessageEncoding,
    inspect_mcap_video,
)

__all__ = [
    "BitstreamFormat",
    "ByteSource",
    "ChannelRecord",
    "ChunkIndexRecord",
    "DirectPlaybackSupport",
    "FileByteSource",
    "FrameKind",
    "HttpRangeByteSource",
    "McapParseError",
    "McapSourceError",
    "McapSummary",
    "McapVideoChannel",
    "McapVideoInspection",
    "MessageEncoding",
    "SchemaRecord",
    "StatisticsRecord",
    "VideoCodec",
    "inspect_mcap_video",
    "parse_h264_sps",
    "read_summary",
    "scan_access_unit",
]
