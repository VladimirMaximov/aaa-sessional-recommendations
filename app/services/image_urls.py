from __future__ import annotations

import logging

from app.core.config import Settings, item_image_key, item_image_url

logger = logging.getLogger(__name__)


class ImageUrlService:
    """Resolves item image URLs (presigned S3 or public base URL)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._s3 = None

        if settings.s3_bucket and settings.s3_access_key_id and settings.s3_secret_access_key:
            import boto3
            from botocore.config import Config

            self._s3 = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.s3_access_key_id,
                aws_secret_access_key=settings.s3_secret_access_key,
                region_name=settings.s3_region,
                config=Config(signature_version="s3"),
            )
            logger.info(
                "Image URLs: presigned S3 (bucket=%s, ttl=%ds)",
                settings.s3_bucket,
                settings.s3_presign_ttl_seconds,
            )
        elif settings.image_base_url:
            logger.info("Image URLs: public base %s", settings.image_base_url)
        else:
            logger.warning("Image URLs: not configured — set S3 credentials or IMAGE_BASE_URL")

    def url_for(self, item_id: int) -> str:
        key = item_image_key(item_id)
        if self._s3 and self._settings.s3_bucket:
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._settings.s3_bucket, "Key": key},
                ExpiresIn=self._settings.s3_presign_ttl_seconds,
            )
        if self._settings.image_base_url:
            return item_image_url(item_id, self._settings.image_base_url)
        return ""
