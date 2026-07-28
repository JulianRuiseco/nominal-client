from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nominal.core import mcap_video as mcap_video_module
from nominal.core._mcap.registration import (
    DirectMcapUnsupportedError,
    S3Source,
    UriSource,
    source_from_uri,
)
from nominal.core._mcap.sources import FileByteSource
from nominal.core.dataset import Dataset, DatasetBounds
from nominal.core.exceptions import NominalError
from nominal.core.mcap_video import McapVideoRegistration, resolve_mcap_sources
from tests.core import mcap_fixtures as fx


@pytest.fixture
def mock_clients():
    clients = MagicMock()
    clients.auth_header = "Bearer test-token"
    clients.direct_mcap.register.return_value = {"channels": []}
    return clients


@pytest.fixture
def dataset(mock_clients):
    return Dataset(
        rid="ri.catalog.main.dataset.test",
        name="Test Dataset",
        description=None,
        bounds=DatasetBounds(start=0, end=1),
        properties={},
        labels=[],
        _clients=mock_clients,
    )


def write_mcap(tmp_path, name="recording.mcap", **kwargs):
    path = tmp_path / name
    path.write_bytes(fx.build_video_mcap(**kwargs))
    return path


@pytest.fixture(autouse=True)
def serve_local_files(monkeypatch, tmp_path):
    """Resolve the https:// URLs used in these tests to files on disk.

    Registration only ever needs ranged reads, so a local file is a faithful stand-in for an object
    in a customer bucket. Local paths still take the real code path, so the check that rejects them
    is genuinely exercised.
    """
    real = mcap_video_module.open_byte_source

    def fake(uri: str, **kwargs):
        if uri.startswith("https://example.test/"):
            return FileByteSource.from_path(str(tmp_path / uri.rsplit("/", 1)[-1]))
        return real(uri, **kwargs)

    monkeypatch.setattr(mcap_video_module, "open_byte_source", fake)


class TestSourceResolution:
    def test_maps_s3_paths_to_a_bucket_and_key(self):
        source = source_from_uri("s3://saronic-recordings/missions/0142/front.mcap")

        assert isinstance(source, S3Source)
        assert source.bucket == "saronic-recordings"
        assert source.key == "missions/0142/front.mcap"
        assert source.to_conjure() == {"s3": {"bucket": "saronic-recordings", "key": "missions/0142/front.mcap"}}

    def test_maps_https_urls_to_a_uri_source(self):
        source = source_from_uri("https://example.test/recording.mcap")

        assert isinstance(source, UriSource)
        assert source.to_conjure() == {"uri": {"url": "https://example.test/recording.mcap"}}

    def test_refuses_a_location_a_browser_cannot_reach(self):
        with pytest.raises(ValueError, match="neither an s3:// path nor an https:// URL"):
            source_from_uri("/local/recording.mcap")

    def test_lists_every_mcap_under_an_s3_prefix(self):
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "missions/0142/b.mcap"}, {"Key": "missions/0142/notes.txt"}]},
            {"Contents": [{"Key": "missions/0142/a.mcap"}]},
        ]

        found = resolve_mcap_sources("s3://recordings/missions/0142/", s3_client=client)

        assert found == ["s3://recordings/missions/0142/a.mcap", "s3://recordings/missions/0142/b.mcap"]

    def test_passes_a_single_object_through_untouched(self):
        assert resolve_mcap_sources("s3://recordings/a.mcap") == ["s3://recordings/a.mcap"]
        assert resolve_mcap_sources("/tmp/a.mcap") == ["/tmp/a.mcap"]


class TestReferenceModeRegistration:
    def test_registers_video_channels_without_uploading_anything(self, dataset, mock_clients, tmp_path):
        path = write_mcap(tmp_path, frame_count=120, keyframe_every=15)
        mock_clients.direct_mcap.register.return_value = {
            "channels": [{"channel": "/camera/front", "topic": "/camera/front", "seriesRid": "ri.series.1"}]
        }

        result = dataset.add_mcap_video(f"https://example.test/{path.name}", copy=False)

        # No upload, no ingest: the whole point.
        mock_clients.upload.assert_not_called()
        mock_clients.ingest.ingest.assert_not_called()
        assert isinstance(result, McapVideoRegistration)
        assert len(result.direct_channels) == 1
        assert result.direct_channels[0].topic == "/camera/front"

    def test_registers_the_topic_as_a_video_series_on_the_dataset(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path, frame_count=60, keyframe_every=10)

        dataset.add_mcap_video("https://example.test/recording.mcap", copy=False, tags={"vehicle": "boat-1"})

        mock_clients.series_metadata.batch_create_video_series.assert_called_once()
        _auth, request = mock_clients.series_metadata.batch_create_video_series.call_args.args
        assert len(request.requests) == 1
        created = request.requests[0]
        assert created.dataset_rid == "ri.catalog.main.dataset.test"
        assert created.channel == "/camera/front"
        assert created.tags == {"vehicle": "boat-1"}
        assert created.time_bounds.end.seconds >= created.time_bounds.start.seconds

    def test_sends_the_files_own_chunk_index_rather_than_deriving_one(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path, frame_count=100, keyframe_every=10, messages_per_chunk=25)

        dataset.add_mcap_video("https://example.test/recording.mcap", copy=False)

        _auth, request = mock_clients.direct_mcap.register.call_args.args
        assert request["datasetRid"] == "ri.catalog.main.dataset.test"
        assert len(request["files"]) == 1
        channel = request["files"][0]["channels"][0]
        assert channel["codec"] == "H264"
        assert channel["bitstreamFormat"] == "ANNEX_B"
        assert channel["codecString"] == "avc1.42c01f"
        assert channel["width"] == 1920
        assert len(channel["chunks"]) == 4
        assert all(chunk["length"] > 0 for chunk in channel["chunks"])

    def test_raises_by_default_when_a_topic_cannot_be_played_directly(self, dataset, tmp_path):
        write_mcap(tmp_path, codec="vp9")

        with pytest.raises(NominalError, match="cannot be played directly"):
            dataset.add_mcap_video("https://example.test/recording.mcap", copy=False)

    def test_routes_unsupported_topics_to_ingest_when_asked(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path, codec="vp9")

        result = dataset.add_mcap_video("https://example.test/recording.mcap", copy=False, on_unsupported="ingest")

        mock_clients.ingest.ingest.assert_called_once()
        _auth, request = mock_clients.ingest.ingest.call_args.args
        assert request.options.video_v2.channel == "/camera/front"
        assert request.options.video_v2.timestamp_manifest.mcap.mcap_channel_locator.topic == "/camera/front"
        assert len(result.ingested_channels) == 1
        assert result.direct_channels == ()

    def test_skips_unsupported_topics_when_asked(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path, codec="vp9")

        result = dataset.add_mcap_video("https://example.test/recording.mcap", copy=False, on_unsupported="skip")

        mock_clients.ingest.ingest.assert_not_called()
        mock_clients.direct_mcap.register.assert_not_called()
        assert len(result.skipped_channels) == 1
        assert "vp9" in result.skipped_channels[0].skipped_reason

    def test_honours_a_channel_name_override(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path)

        dataset.add_mcap_video(
            "https://example.test/recording.mcap",
            copy=False,
            channel_names={"/camera/front": "front_camera"},
        )

        _auth, request = mock_clients.series_metadata.batch_create_video_series.call_args.args
        assert request.requests[0].channel == "front_camera"

    def test_uploads_a_local_path_and_registers_the_uploaded_object(self, dataset, mock_clients, tmp_path, monkeypatch):
        path = write_mcap(tmp_path)
        mock_clients.direct_mcap.register.return_value = {
            "channels": [{"channel": "/camera/front", "topic": "/camera/front", "seriesRid": "ri.series.1"}]
        }
        uploaded = {}

        def fake_upload(auth_header, workspace_rid, upload_path, upload_client, **kwargs):
            uploaded["path"] = upload_path
            return "s3://upload-bucket/uploads/abc123/recording.mcap"

        monkeypatch.setattr(mcap_video_module, "upload_multipart_file", fake_upload)

        result = dataset.add_mcap_video(str(path), copy=False)

        # The local file was inspected in place but the registration points at the uploaded copy.
        assert uploaded["path"] == path
        _auth, request = mock_clients.direct_mcap.register.call_args.args
        file = request["files"][0]
        assert file["source"] == {"s3": {"bucket": "upload-bucket", "key": "uploads/abc123/recording.mcap"}}
        # The multipart etag is not the md5 of the local bytes, so none is pinned.
        assert file["etag"] is None
        assert len(result.direct_channels) == 1

    def test_does_not_upload_a_local_file_with_nothing_playable(self, dataset, mock_clients, tmp_path, monkeypatch):
        path = write_mcap(tmp_path, codec="vp9")
        upload = MagicMock()
        monkeypatch.setattr(mcap_video_module, "upload_multipart_file", upload)

        result = dataset.add_mcap_video(str(path), copy=False, on_unsupported="skip")

        upload.assert_not_called()
        mock_clients.direct_mcap.register.assert_not_called()
        assert result.direct_channels == ()

    def test_reports_how_little_was_read(self, dataset, tmp_path):
        path = write_mcap(tmp_path, frame_count=300, keyframe_every=30, messages_per_chunk=50)

        result = dataset.add_mcap_video("https://example.test/recording.mcap", copy=False)

        assert result.bytes_read > 0
        assert result.bytes_read < path.stat().st_size
        assert "direct" in result.summary()

    def test_surfaces_a_deployment_without_direct_playback(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path)
        mock_clients.direct_mcap.register.side_effect = DirectMcapUnsupportedError("not supported")

        with pytest.raises(DirectMcapUnsupportedError):
            dataset.add_mcap_video("https://example.test/recording.mcap", copy=False)


class TestCopyMode:
    def test_ingests_every_video_topic_when_copying(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path)

        result = dataset.add_mcap_video("https://example.test/recording.mcap", copy=True)

        mock_clients.ingest.ingest.assert_called_once()
        mock_clients.direct_mcap.register.assert_not_called()
        assert len(result.ingested_channels) == 1

    def test_add_mcap_with_copy_false_delegates_to_reference_mode(self, dataset, mock_clients, tmp_path):
        write_mcap(tmp_path)

        result = dataset.add_mcap("https://example.test/recording.mcap", copy=False)

        assert isinstance(result, McapVideoRegistration)
        mock_clients.direct_mcap.register.assert_called_once()
