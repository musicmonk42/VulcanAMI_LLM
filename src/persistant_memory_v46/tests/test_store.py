"""
Comprehensive tests for store.py module.

Tests cover:
- S3Store functionality
- PackfileStore operations
- Upload/download with compression and encryption
- Caching mechanisms
- CloudFront integration
- Async operations
- Edge cases
"""

import asyncio
import hashlib
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

sys.path.insert(0, "/mnt/user-data/uploads")

from store import PackfileStore, S3Store


class TestS3Store:
    """Test suite for S3Store class."""

    def test_initialization(self):
        """Test S3Store initialization."""
        store = S3Store(bucket="test-bucket", region="us-east-1")

        assert store.bucket == "test-bucket"
        assert store.region == "us-east-1"
        assert store.endpoint == "https://s3.us-east-1.amazonaws.com"
        assert store.use_ssl is True

    def test_custom_endpoint(self):
        """Test S3Store with custom endpoint."""
        store = S3Store(bucket="test-bucket", endpoint="https://custom-s3.example.com")

        assert store.endpoint == "https://custom-s3.example.com"

    @patch("requests.Session.put")
    def test_put_object(self, mock_put):
        """Test putting an object to S3."""
        # Setup mock
        mock_response = Mock()
        mock_response.headers = {"ETag": '"abc123"', "x-amz-version-id": "v1"}
        mock_response.raise_for_status = Mock()
        mock_put.return_value = mock_response

        store = S3Store(bucket="test-bucket")

        result = store.put_object(
            key="test-key", data=b"test data", encryption="AES256"
        )

        assert result["key"] == "test-key"
        assert result["etag"] == "abc123"
        assert result["version_id"] == "v1"
        assert result["size"] == 9

    @patch("requests.Session.get")
    def test_get_object(self, mock_get):
        """Test getting an object from S3."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = b"test data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        store = S3Store(bucket="test-bucket")
        data = store.get_object(key="test-key")

        assert data == b"test data"

    @patch("requests.Session.get")
    def test_get_object_with_range(self, mock_get):
        """Test getting an object with byte range."""
        mock_response = Mock()
        mock_response.content = b"partial"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        store = S3Store(bucket="test-bucket")
        data = store.get_object(key="test-key", byte_range=(0, 100))

        assert data == b"partial"
        # Verify Range header was set
        call_args = mock_get.call_args
        assert "Range" in call_args[1]["headers"]

    @patch("requests.Session.delete")
    def test_delete_object(self, mock_delete):
        """Test deleting an object from S3."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response

        store = S3Store(bucket="test-bucket")

        # Should not raise
        store.delete_object(key="test-key")

    @patch("requests.Session.get")
    def test_list_objects(self, mock_get):
        """Test listing objects in S3."""
        mock_response = Mock()
        mock_response.text = """
        <ListBucketResult>
            <Contents>
                <Key>file1.txt</Key>
                <Size>100</Size>
            </Contents>
            <Contents>
                <Key>file2.txt</Key>
                <Size>200</Size>
            </Contents>
            <IsTruncated>false</IsTruncated>
        </ListBucketResult>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        store = S3Store(bucket="test-bucket")
        result = store.list_objects(prefix="test/")

        assert "objects" in result
        assert len(result["objects"]) == 2
        assert result["objects"][0]["key"] == "file1.txt"
        assert result["objects"][0]["size"] == 100

    def test_sign_request(self):
        """Test request signing."""
        store = S3Store(
            bucket="test-bucket",
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )

        headers = store._sign_request("GET", "test-key", {})

        # Should add authorization headers
        assert isinstance(headers, dict)


class TestPackfileStore:
    """Test suite for PackfileStore class."""

    def test_initialization(self):
        """Test PackfileStore initialization."""
        with patch("store.S3Store"):
            store = PackfileStore(
                s3_bucket="test-bucket", compression="zstd", encryption="AES256"
            )

            assert store.compression == "zstd"
            assert store.encryption == "AES256"
            assert store.storage_class == "INTELLIGENT_TIERING"
            assert store.cache == {}

    def test_initialization_with_zstd(self):
        """Test initialization with zstd compression."""
        with patch("store.S3Store"):
            try:
                import zstandard

                store = PackfileStore(s3_bucket="test-bucket", compression="zstd")
                assert store.compression == "zstd"
            except ImportError:
                pytest.skip("zstandard not installed")

    def test_initialization_with_lz4(self):
        """Test initialization with lz4 compression."""
        with patch("store.S3Store"):
            try:
                import lz4.frame

                store = PackfileStore(s3_bucket="test-bucket", compression="lz4")
                assert store.compression == "lz4"
            except ImportError:
                pytest.skip("lz4 not installed")

    @patch("store.S3Store")
    def test_upload(self, mock_s3):
        """Test packfile upload."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc123", "size": 100}
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        pack_bytes = b"test packfile data"
        path = store.upload(pack_bytes, level=0)

        assert isinstance(path, str)
        assert "packs/" in path
        assert mock_s3_instance.put_object.called

    @patch("store.S3Store")
    def test_upload_with_compression(self, mock_s3):
        """Test upload with compression."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 50}
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket", compression="zlib")

        pack_bytes = b"a" * 1000  # Compressible data
        path = store.upload(pack_bytes)

        # Verify compression was applied
        call_args = mock_s3_instance.put_object.call_args
        compressed_data = call_args[1]["data"]
        assert len(compressed_data) < len(pack_bytes)

    @patch("store.S3Store")
    def test_download(self, mock_s3):
        """Test packfile download."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.return_value = b"test data"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        data = store.download("packs/test.pack")

        assert data == b"test data"
        assert mock_s3_instance.get_object.called

    @patch("store.S3Store")
    def test_download_with_cache(self, mock_s3):
        """Test download with caching."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.return_value = b"cached data"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket", enable_cache=True)

        path = "packs/test.pack"

        # First download - should hit S3
        data1 = store.download(path, use_cache=True)
        assert mock_s3_instance.get_object.call_count == 1

        # Second download - should use cache
        data2 = store.download(path, use_cache=True)
        assert mock_s3_instance.get_object.call_count == 1  # No additional call

        assert data1 == data2

    @patch("store.S3Store")
    def test_download_with_byte_range(self, mock_s3):
        """Test download with byte range."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.return_value = b"partial"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        data = store.download("packs/test.pack", byte_range=(0, 99))

        assert data == b"partial"

    @patch("store.S3Store")
    def test_delete(self, mock_s3):
        """Test packfile deletion."""
        mock_s3_instance = Mock()
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        # Add to cache first
        store.cache["packs/test.pack"] = b"data"

        store.delete("packs/test.pack")

        # Should be removed from cache
        assert "packs/test.pack" not in store.cache
        assert mock_s3_instance.delete_object.called

    @patch("store.S3Store")
    def test_list_packfiles(self, mock_s3):
        """Test listing packfiles."""
        mock_s3_instance = Mock()
        mock_s3_instance.list_objects.return_value = {
            "objects": [
                {"key": "packs/pack1.pack", "size": 100},
                {"key": "packs/pack2.pack", "size": 200},
            ]
        }
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        packfiles = store.list_packfiles()

        assert len(packfiles) == 2
        assert packfiles[0]["key"] == "packs/pack1.pack"

    @patch("store.S3Store")
    def test_compression_decompression(self, mock_s3):
        """Test compression and decompression cycle."""
        mock_s3.return_value = Mock()

        store = PackfileStore(s3_bucket="test-bucket", compression="zlib")

        original_data = b"test data " * 100

        compressed = store._compress(original_data)
        decompressed = store._decompress(compressed)

        assert len(compressed) < len(original_data)
        assert decompressed == original_data

    @patch("store.S3Store")
    def test_upload_verification(self, mock_s3):
        """Test upload verification."""
        mock_s3_instance = Mock()
        test_data = b"test data"
        mock_s3_instance.put_object.return_value = {
            "etag": "abc",
            "size": len(test_data),
        }
        mock_s3_instance.get_object.return_value = test_data
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket", verify_uploads=True)

        path = store.upload(test_data)

        # Should not raise
        assert path is not None

    @patch("store.S3Store")
    @patch("requests.get")
    def test_cloudfront_download(self, mock_get, mock_s3):
        """Test download from CloudFront."""
        mock_response = Mock()
        mock_response.content = b"cloudfront data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        mock_s3.return_value = Mock()

        store = PackfileStore(
            s3_bucket="test-bucket", cloudfront_url="https://d123.cloudfront.net"
        )

        data = store.download("packs/test.pack")

        assert data == b"cloudfront data"
        assert mock_get.called

    @patch("store.S3Store")
    def test_adaptive_range_download(self, mock_s3):
        """Test adaptive range download."""
        mock_s3_instance = Mock()
        # First chunk
        mock_s3_instance.get_object.side_effect = [
            b"a" * (1024 * 1024),  # 1MB first chunk
            b"b" * 1000,  # Remaining data
        ]
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket", enable_adaptive_range=True)

        data = store._adaptive_range_download("packs/test.pack")

        assert len(data) > 1024 * 1024

    @patch("store.S3Store")
    def test_cache_eviction(self, mock_s3):
        """Test cache LRU eviction."""
        mock_s3.return_value = Mock()

        store = PackfileStore(
            s3_bucket="test-bucket",
            enable_cache=True,
            cache_max_size_mb=1,  # 1MB cache
        )

        # Add data larger than cache
        large_data = b"x" * (2 * 1024 * 1024)  # 2MB
        store._add_to_cache("test1", large_data[: 1024 * 1024])
        store._add_to_cache("test2", large_data[: 1024 * 1024])

        # Should evict oldest entry
        assert len(store.cache) <= 1

    @patch("store.S3Store")
    def test_prefetch(self, mock_s3):
        """Test prefetching packfiles."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.return_value = b"prefetch data"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(
            s3_bucket="test-bucket", enable_cache=True, prefetch_enabled=True
        )

        paths = ["packs/pack1.pack", "packs/pack2.pack"]
        store.prefetch(paths)

        # Should be in cache now
        assert len(store.cache) > 0

    @patch("store.S3Store")
    @pytest.mark.asyncio
    async def test_async_upload(self, mock_s3):
        """Test async upload."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 10}
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        path = await store.upload_async(b"test data")

        assert isinstance(path, str)

    @patch("store.S3Store")
    @pytest.mark.asyncio
    async def test_async_download(self, mock_s3):
        """Test async download."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.return_value = b"async data"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        data = await store.download_async("packs/test.pack")

        assert data == b"async data"

    @patch("store.S3Store")
    def test_get_statistics(self, mock_s3):
        """Test statistics retrieval."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 10}
        mock_s3_instance.get_object.return_value = b"data"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        # Perform operations
        store.upload(b"data")
        store.download("packs/test.pack")

        stats = store.get_statistics()

        assert "upload_stats" in stats
        assert "download_stats" in stats
        assert "cache_size_mb" in stats
        assert "compression" in stats

    @patch("store.S3Store")
    def test_clear_cache(self, mock_s3):
        """Test cache clearing."""
        mock_s3.return_value = Mock()

        store = PackfileStore(s3_bucket="test-bucket")

        # Add to cache
        store.cache["test"] = b"data"
        store.cache_current_size = 4

        store.clear_cache()

        assert len(store.cache) == 0
        assert store.cache_current_size == 0

    @patch("store.S3Store")
    def test_close(self, mock_s3):
        """Test resource cleanup."""
        mock_s3.return_value = Mock()

        store = PackfileStore(s3_bucket="test-bucket")

        # Should not raise
        store.close()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("store.S3Store")
    def test_empty_data_upload(self, mock_s3):
        """Test uploading empty data."""
        mock_s3_instance = Mock()
        mock_s3_instance.put_object.return_value = {"etag": "abc", "size": 0}
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        path = store.upload(b"")

        assert isinstance(path, str)

    @patch("store.S3Store")
    def test_large_file_upload(self, mock_s3):
        """Test uploading large file."""
        mock_s3_instance = Mock()
        mock_s3_instance.multipart_upload.return_value = "packs/large.pack"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        # Create 100MB file
        large_data = b"x" * (100 * 1024 * 1024)

        # Should handle large files
        path = store.upload(large_data)
        assert isinstance(path, str)

    @patch("store.S3Store")
    def test_download_nonexistent_file(self, mock_s3):
        """Test downloading non-existent file."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.side_effect = Exception("Not found")
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket")

        with pytest.raises(Exception):
            store.download("packs/nonexistent.pack")

    @patch("store.S3Store")
    def test_corrupted_compressed_data(self, mock_s3):
        """Test handling corrupted compressed data."""
        mock_s3_instance = Mock()
        mock_s3_instance.get_object.return_value = b"corrupted"
        mock_s3.return_value = mock_s3_instance

        store = PackfileStore(s3_bucket="test-bucket", compression="zlib")

        with pytest.raises(Exception):
            store.download("packs/corrupted.pack", decompress=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
