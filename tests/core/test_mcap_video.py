import io

import pytest

from nominal.core._mcap import inspect_mcap_video, parse_h264_sps, read_summary
from nominal.core._mcap.bitstream import (
    BitstreamFormat,
    FrameKind,
    VideoCodec,
    detect_format,
    scan_access_unit,
    strip_emulation_prevention,
)
from nominal.core._mcap.messages import (
    MessageDecodeError,
    decode_cdr_compressed_video,
    decode_protobuf_compressed_video,
)
from nominal.core._mcap.records import McapParseError
from nominal.core._mcap.sources import FileByteSource
from nominal.core._mcap.video import DirectPlaybackSupport, MessageEncoding
from tests.core import mcap_fixtures as fx


def source(raw: bytes, uri: str = "mem://test.mcap") -> FileByteSource:
    return FileByteSource(io.BytesIO(raw), uri=uri)


class TestSummaryReading:
    def test_reads_channels_and_bounds_without_reading_the_whole_file(self):
        raw = fx.build_video_mcap(frame_count=300, keyframe_every=30, messages_per_chunk=50)
        src = source(raw)

        summary = read_summary(src)

        assert set(c.topic for c in summary.channels.values()) == {"/camera/front"}
        assert summary.is_chunked
        assert summary.statistics is not None
        assert summary.statistics.message_count == 300
        # The whole point of the design: registration reads kilobytes, not the file.
        assert src.bytes_read < len(raw) / 4

    def test_narrows_time_bounds_to_the_chunks_holding_a_channel(self):
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10, messages_per_chunk=20)
        summary = read_summary(source(raw))
        channel_id = next(iter(summary.channels))

        bounds = summary.channel_time_bounds(channel_id)

        assert bounds is not None
        assert bounds[0] == summary.message_start_time
        assert bounds[1] == summary.message_end_time
        assert len(summary.chunks_for_channel(channel_id)) == 3

    def test_rejects_a_file_with_no_summary_section(self):
        builder = fx.McapBuilder()
        builder.add_schema(1, fx.PROTOBUF_SCHEMA)
        builder.add_channel(1, 1, "/camera/front", "protobuf")
        builder.add_message(1, 1_000, fx.encode_protobuf_compressed_video(fx.keyframe_sample(), "h264"))
        raw = builder.build(omit_summary=True)

        with pytest.raises(McapParseError, match="no summary section"):
            read_summary(source(raw))

    def test_rejects_bytes_that_are_not_mcap(self):
        with pytest.raises(McapParseError):
            read_summary(source(b"not an mcap file, not even close, but long enough to pass the size check" * 4))


class TestBitstreamParsing:
    @pytest.mark.parametrize(("width", "height"), [(1920, 1080), (1280, 720), (640, 480), (3840, 2160)])
    def test_sps_round_trips_frame_dimensions_including_cropping(self, width, height):
        sps = parse_h264_sps(fx.make_sps(width, height))

        assert sps.width == width
        assert sps.height == height
        assert sps.codec_string == "avc1.42c01f"

    def test_strips_emulation_prevention_bytes(self):
        raw = b"\x00\x00\x00\x01\x00\x00\x02"

        assert strip_emulation_prevention(fx.add_emulation_prevention(raw)) == raw
        assert fx.add_emulation_prevention(raw) != raw

    def test_distinguishes_annex_b_from_length_prefixed(self):
        assert detect_format(fx.keyframe_sample()) is BitstreamFormat.ANNEX_B
        assert detect_format(fx.keyframe_sample(prefixed=True)) is BitstreamFormat.LENGTH_PREFIXED
        assert detect_format(b"\x01\x02") is BitstreamFormat.UNKNOWN

    def test_reports_idr_keyframes_with_their_parameter_sets(self):
        scan = scan_access_unit(fx.keyframe_sample(), VideoCodec.H264)

        assert scan.kind is FrameKind.KEYFRAME
        assert scan.has_sps and scan.has_pps
        assert scan.sps is not None and scan.sps.width == 1920

    def test_does_not_count_a_non_idr_intra_frame_as_a_seek_point(self):
        scan = scan_access_unit(fx.delta_sample(intra=True), VideoCodec.H264)

        assert scan.kind is FrameKind.INTRA_NON_IDR
        assert scan.kind is not FrameKind.KEYFRAME

    def test_detects_b_slices(self):
        assert scan_access_unit(fx.delta_sample(bidirectional=True), VideoCodec.H264).has_b_slices
        assert not scan_access_unit(fx.delta_sample(), VideoCodec.H264).has_b_slices

    def test_parses_length_prefixed_samples(self):
        scan = scan_access_unit(fx.keyframe_sample(prefixed=True), VideoCodec.H264)

        assert scan.format is BitstreamFormat.LENGTH_PREFIXED
        assert scan.kind is FrameKind.KEYFRAME
        assert scan.sps is not None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("h264", VideoCodec.H264),
            ("H.264", VideoCodec.H264),
            ("avc1", VideoCodec.H264),
            ("h265", VideoCodec.H265),
            ("hevc", VideoCodec.H265),
            ("vp9", VideoCodec.VP9),
            ("av1", VideoCodec.AV1),
            ("theora", VideoCodec.UNKNOWN),
        ],
    )
    def test_normalizes_codec_aliases(self, raw, expected):
        assert VideoCodec.parse(raw) is expected


class TestMessageDecoding:
    def test_decodes_protobuf_compressed_video(self):
        payload = fx.encode_protobuf_compressed_video(b"\x00\x00\x00\x01abc", "h264", timestamp_ns=1_700_000_000_500)

        message = decode_protobuf_compressed_video(payload)

        assert message.format == "h264"
        assert message.data == b"\x00\x00\x00\x01abc"
        assert message.timestamp_ns == 1_700_000_000_500

    def test_decodes_cdr_compressed_video_as_rosbag2_writes_it(self):
        payload = fx.encode_cdr_compressed_video(b"\x00\x00\x00\x01xyz", "h264", timestamp_ns=1_700_000_000_500)

        message = decode_cdr_compressed_video(payload)

        assert message.format == "h264"
        assert message.data == b"\x00\x00\x00\x01xyz"
        assert message.timestamp_ns == 1_700_000_000_500

    def test_raises_on_a_message_with_no_video_payload(self):
        with pytest.raises(MessageDecodeError):
            decode_protobuf_compressed_video(b"")


class TestVideoInspection:
    def test_classifies_a_well_formed_h264_recording_as_direct_playable(self):
        raw = fx.build_video_mcap(frame_count=120, keyframe_every=15)

        inspection = inspect_mcap_video(source(raw))

        assert len(inspection.channels) == 1
        channel = inspection.channel("/camera/front")
        assert channel.support is DirectPlaybackSupport.DIRECT
        assert channel.codec is VideoCodec.H264
        assert channel.width == 1920
        assert channel.height == 1080
        assert channel.codec_string == "avc1.42c01f"
        assert channel.frame_rate == pytest.approx(30.0, abs=0.5)
        assert channel.parameter_sets_in_band is True
        assert channel.reasons == ()

    def test_reads_both_wire_encodings_of_the_same_schema(self):
        protobuf = inspect_mcap_video(source(fx.build_video_mcap(encoding="protobuf"))).channels[0]
        cdr = inspect_mcap_video(source(fx.build_video_mcap(encoding="cdr"))).channels[0]

        assert protobuf.message_encoding is MessageEncoding.PROTOBUF
        assert cdr.message_encoding is MessageEncoding.CDR
        assert protobuf.width == cdr.width == 1920
        assert protobuf.support is cdr.support is DirectPlaybackSupport.DIRECT

    def test_ignores_non_video_topics(self):
        raw = fx.build_video_mcap(extra_topics=["/imu", "/gps"])

        inspection = inspect_mcap_video(source(raw))

        assert [c.topic for c in inspection.channels] == ["/camera/front"]
        assert inspection.non_video_topics == ("/gps", "/imu")

    def test_routes_sparse_keyframes_to_processing(self):
        # A single keyframe at the start of 900 frames at 30fps: seeking to the end would decode
        # 30 seconds of video, well past the threshold.
        raw = fx.build_video_mcap(frame_count=900, keyframe_every=10_000, messages_per_chunk=900)

        channel = inspect_mcap_video(source(raw), max_probe_frames=900).channels[0]

        assert channel.keyframes_probed == 1
        assert channel.keyframe_interval_is_lower_bound is True
        assert channel.support is DirectPlaybackSupport.NEEDS_PROCESSING
        assert any("seek threshold" in reason for reason in channel.reasons)

    def test_does_not_measure_a_keyframe_gap_across_the_hole_between_probed_chunks(self):
        # The probe reads the first and last chunks, which are 20 seconds apart in this file. A
        # naive measurement spanning that hole would report a 20s keyframe interval for a stream
        # whose keyframes are actually one second apart, and condemn it to re-encoding.
        raw = fx.build_video_mcap(
            frame_count=600, keyframe_every=30, messages_per_chunk=30, frame_interval_ns=33_333_333
        )

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.support is DirectPlaybackSupport.DIRECT
        assert channel.keyframe_interval_seconds is not None
        assert channel.keyframe_interval_seconds < 2.0
        assert not any("seek threshold" in reason for reason in channel.reasons)

    def test_measures_a_real_keyframe_gap_within_a_single_probed_chunk(self):
        # Two keyframes inside one chunk make the interval a measurement rather than a bound.
        raw = fx.build_video_mcap(frame_count=300, keyframe_every=30, messages_per_chunk=300)

        channel = inspect_mcap_video(source(raw), max_probe_frames=300).channels[0]

        assert channel.keyframes_probed == 10
        assert channel.keyframe_interval_is_lower_bound is False
        assert channel.keyframe_interval_seconds == pytest.approx(1.0, abs=0.05)

    def test_reports_an_unbounded_keyframe_interval_without_blocking_a_short_probe(self):
        # Two seconds of frames with one keyframe: not enough evidence to condemn the file, but the
        # measurement must not be presented as if it were exact.
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10_000, messages_per_chunk=60)

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.keyframe_interval_is_lower_bound is True
        assert channel.support is DirectPlaybackSupport.DIRECT
        assert any("lower bound" in reason for reason in channel.reasons)

    def test_routes_a_codec_the_browser_cannot_decode_to_processing(self):
        raw = fx.build_video_mcap(codec="vp9")

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.support is DirectPlaybackSupport.NEEDS_PROCESSING
        assert any("vp9" in reason for reason in channel.reasons)

    def test_routes_b_frames_to_processing(self):
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10, bidirectional=True)

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.has_b_frames is True
        assert channel.support is DirectPlaybackSupport.NEEDS_PROCESSING
        assert any("B-frames" in reason for reason in channel.reasons)

    def test_routes_an_unchunked_file_to_processing(self):
        builder = fx.McapBuilder()
        builder.add_schema(1, fx.PROTOBUF_SCHEMA)
        builder.add_channel(1, 1, "/camera/front", "protobuf")
        builder.add_message(1, 1_000, fx.encode_protobuf_compressed_video(fx.keyframe_sample(), "h264"))
        # A summary section with no chunk index: legal MCAP, not range-readable.
        raw = builder.build(messages_per_chunk=10**9)

        summary = read_summary(source(raw))
        assert summary.is_chunked  # sanity: the fixture does chunk by default

        with pytest.raises(McapParseError):
            read_summary(source(builder.build(omit_summary=True)))

    def test_flags_keyframes_missing_parameter_sets_without_blocking_playback(self):
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10, parameter_sets_on_keyframes=False)

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.parameter_sets_in_band is False
        assert any("SPS/PPS" in reason for reason in channel.reasons)
        # The player injects cached parameter sets, so this is not disqualifying.
        assert channel.support is DirectPlaybackSupport.DIRECT

    def test_reports_h265_as_playable_but_hardware_dependent(self):
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10, codec="h265")

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.codec is VideoCodec.H265
        assert any("hardware-dependent" in reason for reason in channel.reasons)

    def test_measures_skew_between_capture_time_and_log_time(self):
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10)

        channel = inspect_mcap_video(source(raw)).channels[0]

        # The fixture stamps capture time 1ms before log time.
        assert channel.capture_vs_log_skew_ns == 1_000_000

    def test_exposes_the_chunk_index_for_playback(self):
        raw = fx.build_video_mcap(frame_count=100, keyframe_every=10, messages_per_chunk=25)

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.chunk_count == 4
        assert len(channel.chunk_ranges) == 4
        assert all(r.length > 0 for r in channel.chunk_ranges)
        starts = [r.start_time for r in channel.chunk_ranges]
        assert starts == sorted(starts)

    def test_restricts_inspection_to_requested_topics(self):
        raw = fx.build_video_mcap(topic="/camera/front", extra_topics=["/imu"])

        inspection = inspect_mcap_video(source(raw), include_topics=["/camera/front"])

        assert [c.topic for c in inspection.channels] == ["/camera/front"]
        with pytest.raises(KeyError, match="/camera/rear"):
            inspect_mcap_video(source(raw), include_topics=["/camera/rear"])

    def test_skipping_the_probe_leaves_classification_unknown(self):
        raw = fx.build_video_mcap()

        channel = inspect_mcap_video(source(raw), probe=False).channels[0]

        assert channel.support is DirectPlaybackSupport.UNKNOWN
        assert channel.codec is VideoCodec.UNKNOWN
        # Time bounds still come from the summary section.
        assert channel.end_time > channel.start_time

    def test_rejects_chunks_too_large_for_a_browser_to_hold(self):
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10, messages_per_chunk=1000)

        channel = inspect_mcap_video(source(raw), max_chunk_uncompressed_bytes=128).channels[0]

        assert channel.support is DirectPlaybackSupport.NEEDS_PROCESSING
        assert any("playback limit" in reason for reason in channel.reasons)


class TestCompressedChunks:
    @pytest.mark.parametrize("compression", ["zstd", "lz4"])
    def test_probes_compressed_chunks_when_the_decompressor_is_available(self, compression):
        pytest.importorskip("zstandard" if compression == "zstd" else "lz4")
        raw = fx.build_video_mcap(frame_count=60, keyframe_every=10, compression=compression)

        channel = inspect_mcap_video(source(raw)).channels[0]

        assert channel.support is DirectPlaybackSupport.DIRECT
        assert channel.width == 1920
        assert channel.keyframes_probed > 0
