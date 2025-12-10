"""
S3 Storage Interface

This module provides comprehensive S3 operations with support for multipart uploads,
retry logic, presigned URLs, lifecycle management, and object versioning.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class S3Object:
    """Metadata for an S3 object"""

    key: str
    size: int
    etag: str
    last_modified: Any
    storage_class: str


@dataclass
class UploadResult:
    """Result of an upload operation"""

    key: str
    etag: str
    version_id: Optional[str]
    size: int


class S3Error(Exception):
    """Base exception for S3 operations"""

    pass


class S3Store:
    """
    S3 storage interface with comprehensive operations.

    Features:
    - Get/put operations with range requests
    - Multipart uploads for large files
    - Retry logic with exponential backoff
    - Presigned URLs
    - Object listing and filtering
    - Lifecycle management helpers

    Example:
        store = S3Store(bucket="my-bucket", prefix="data")

        # Upload
        store.put_object("file.dat", b"data...")

        # Download
        data, meta = store.get_object("file.dat")

        # Range request
        data, meta = store.get_object_range("file.dat", (0, 1024))
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize S3 store.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all operations
            region: AWS region (None = default)
            max_retries: Maximum retry attempts
            retry_backoff: Backoff multiplier for retries
        """
        endpoint = os.environ.get("S3_ENDPOINT_URL")

        if endpoint:
            self.s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region)
        else:
            self.s3 = boto3.client("s3", region_name=region)

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        logger.info(f"Initialized S3Store: bucket={bucket}, prefix={prefix}")

    def _key(self, path: str) -> str:
        """Build full S3 key with prefix"""
        path = path.strip("/")
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def _retry_operation(self, operation, *args, **kwargs) -> Any:
        """Execute operation with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                # Don't retry certain errors
                if error_code in ["NoSuchKey", "NoSuchBucket"]:
                    raise

                last_error = e
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_backoff**attempt
                    logger.warning(
                        f"S3 operation failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_backoff**attempt
                    logger.error(
                        f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)

        raise S3Error(
            f"Operation failed after {self.max_retries} attempts"
        ) from last_error

    def get_object_range(
        self, path: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get object with optional byte range.

        Args:
            path: Object path
            byte_range: Optional (start, end) byte range (inclusive)

        Returns:
            Tuple of (data, metadata)
        """
        key = self._key(path)
        kwargs = {"Bucket": self.bucket, "Key": key}

        if byte_range is not None:
            start, end = byte_range
            kwargs["Range"] = f"bytes={start}-{end - 1}"

        def _get():
            resp = self.s3.get_object(**kwargs)
            body = resp["Body"].read()
            return body, resp

        return self._retry_operation(_get)

    def get_object(self, path: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Get entire object.

        Args:
            path: Object path

        Returns:
            Tuple of (data, metadata)
        """
        return self.get_object_range(path, byte_range=None)

    def put_object(
        self,
        path: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        storage_class: str = "STANDARD",
    ) -> UploadResult:
        """
        Put object in S3.

        Args:
            path: Object path
            data: Data to upload
            content_type: Content type
            metadata: Optional user metadata
            storage_class: S3 storage class

        Returns:
            UploadResult with details
        """
        key = self._key(path)

        kwargs = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
            "ContentType": content_type,
            "StorageClass": storage_class,
        }

        if metadata:
            kwargs["Metadata"] = metadata

        def _put():
            resp = self.s3.put_object(**kwargs)
            return UploadResult(
                key=key,
                etag=resp["ETag"].strip('"'),
                version_id=resp.get("VersionId"),
                size=len(data),
            )

        return self._retry_operation(_put)

    def put_object_multipart(
        self,
        path: str,
        file_path: Path,
        part_size: int = 100 * 1024 * 1024,  # 100 MB
        content_type: str = "application/octet-stream",
    ) -> UploadResult:
        """
        Upload large file using multipart upload.

        Args:
            path: Object path
            file_path: Local file to upload
            part_size: Size of each part in bytes
            content_type: Content type

        Returns:
            UploadResult with details
        """
        key = self._key(path)
        file_size = file_path.stat().st_size

        logger.info(f"Starting multipart upload: {key} ({file_size} bytes)")

        # Initiate multipart upload
        mpu = self.s3.create_multipart_upload(
            Bucket=self.bucket, Key=key, ContentType=content_type
        )
        upload_id = mpu["UploadId"]

        parts = []
        part_num = 1

        try:
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(part_size)
                    if not data:
                        break

                    # Upload part
                    part_resp = self.s3.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        PartNumber=part_num,
                        UploadId=upload_id,
                        Body=data,
                    )

                    parts.append({"PartNumber": part_num, "ETag": part_resp["ETag"]})

                    logger.debug(f"Uploaded part {part_num} ({len(data)} bytes)")
                    part_num += 1

            # Complete multipart upload
            result = self.s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(f"Multipart upload completed: {key}")

            return UploadResult(
                key=key,
                etag=result["ETag"].strip('"'),
                version_id=result.get("VersionId"),
                size=file_size,
            )

        except Exception as e:
            # Abort multipart upload on failure
            logger.error(f"Multipart upload failed: {e}")
            try:
                self.s3.abort_multipart_upload(
                    Bucket=self.bucket, Key=key, UploadId=upload_id
                )
            except Exception:
                pass
            raise

    def delete_object(self, path: str) -> bool:
        """
        Delete object.

        Args:
            path: Object path

        Returns:
            True if deleted
        """
        key = self._key(path)

        def _delete():
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True

        return self._retry_operation(_delete)

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[S3Object]:
        """
        List objects with prefix.

        Args:
            prefix: Additional prefix filter
            max_keys: Maximum keys to return

        Returns:
            List of S3Object instances
        """
        full_prefix = self._key(prefix) if prefix else self.prefix

        paginator = self.s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket, Prefix=full_prefix, MaxKeys=max_keys
        )

        objects = []
        for page in pages:
            for obj in page.get("Contents", []):
                objects.append(
                    S3Object(
                        key=obj["Key"],
                        size=obj["Size"],
                        etag=obj["ETag"].strip('"'),
                        last_modified=obj["LastModified"],
                        storage_class=obj.get("StorageClass", "STANDARD"),
                    )
                )

        return objects

    def object_exists(self, path: str) -> bool:
        """Check if object exists"""
        key = self._key(path)

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_object_metadata(self, path: str) -> Dict[str, Any]:
        """Get object metadata without downloading"""
        key = self._key(path)

        resp = self.s3.head_object(Bucket=self.bucket, Key=key)
        return {
            "content_length": resp["ContentLength"],
            "content_type": resp.get("ContentType"),
            "etag": resp["ETag"].strip('"'),
            "last_modified": resp["LastModified"],
            "metadata": resp.get("Metadata", {}),
            "storage_class": resp.get("StorageClass"),
        }

    def generate_presigned_url(
        self, path: str, expiration: int = 3600, method: str = "get_object"
    ) -> str:
        """
        Generate presigned URL for object access.

        Args:
            path: Object path
            expiration: URL expiration in seconds
            method: S3 method (get_object, put_object, etc.)

        Returns:
            Presigned URL
        """
        key = self._key(path)

        url = self.s3.generate_presigned_url(
            method, Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expiration
        )

        return url

    def copy_object(
        self, source_path: str, dest_path: str, source_bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Copy object within or between buckets.

        Args:
            source_path: Source object path
            dest_path: Destination object path
            source_bucket: Source bucket (None = same bucket)

        Returns:
            Copy response
        """
        source_bucket = source_bucket or self.bucket
        source_key = self._key(source_path)
        dest_key = self._key(dest_path)

        copy_source = {"Bucket": source_bucket, "Key": source_key}

        return self.s3.copy_object(
            CopySource=copy_source, Bucket=self.bucket, Key=dest_key
        )

    @staticmethod
    def archive_pack_path(pack_id: str) -> str:
        """Generate S3 path for archive pack"""
        prefix = pack_id[:2]
        return f"archive/packs/{prefix}/{pack_id}.gpk.zst"

    @staticmethod
    def origin_memory_path(content_hash: str) -> str:
        """Generate S3 path for origin memory"""
        # Extract hash suffix (after :)
        hash_value = content_hash.split(":")[-1]
        prefix = hash_value[:2]
        return f"origin/v1/memory/{prefix}/{content_hash}.gpk.zst"

    @staticmethod
    def vector_index_path(tier: str, shard_id: int) -> str:
        """Generate S3 path for vector index shard"""
        return f"vectors/{tier}/shard_{shard_id:04d}.idx"

    def get_bucket_size(self) -> int:
        """Get total size of objects in bucket with prefix"""
        total_size = 0

        for obj in self.list_objects():
            total_size += obj.size

        return total_size
