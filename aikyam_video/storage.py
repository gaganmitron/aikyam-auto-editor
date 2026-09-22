"""ObjectStorage abstraction (section 17). S3-compatible: AWS S3, R2, Linode, MinIO. Local for dev/tests."""
from __future__ import annotations
import os, shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class ObjectStorage(ABC):
    @abstractmethod
    def upload(self, local_path: str, key: str, content_type: Optional[str] = None) -> str: ...
    @abstractmethod
    def download(self, key: str, local_path: str) -> str: ...
    @abstractmethod
    def signed_url(self, key: str, expires: int = 3600, upload: bool = False) -> str: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalStorage(ObjectStorage):
    def __init__(self, root: str):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("key escapes storage root")
        return p

    def upload(self, local_path, key, content_type=None):
        p = self._p(key); p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(local_path, p); return key

    def download(self, key, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(self._p(key), local_path); return local_path

    def signed_url(self, key, expires=3600, upload=False):
        return self._p(key).as_uri()  # dev only: not actually signed

    def delete(self, key):
        self._p(key).unlink(missing_ok=True)

    def exists(self, key):
        return self._p(key).exists()


class S3Storage(ObjectStorage):
    def __init__(self, bucket: str, endpoint_url: Optional[str] = None, region: str = "auto",
                 access_key: Optional[str] = None, secret_key: Optional[str] = None, client=None,
                 public_endpoint_url: Optional[str] = None):
        """public_endpoint_url: the host clients/browsers reach (e.g. the CDN/R2 domain). SigV4 signs the host, so signed URLs must be
        generated for THIS host; it differs from `endpoint_url` when the app talks to storage over an internal address."""
        import boto3
        from botocore.config import Config
        self.bucket = bucket
        mk = lambda ep: boto3.client("s3", endpoint_url=ep, region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                     config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
        self.s3 = client or mk(endpoint_url)
        self.signer = mk(public_endpoint_url) if public_endpoint_url else self.s3   # presigning is a local computation: no network call

    def upload(self, local_path, key, content_type=None):
        extra = {"ContentType": content_type} if content_type else None
        self.s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra)
        return key

    def download(self, key, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path); return local_path

    def signed_url(self, key, expires=3600, upload=False):
        return self.signer.generate_presigned_url("put_object" if upload else "get_object",
                                              Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires)

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key):
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key); return True
        except ClientError:
            return False


def from_env() -> ObjectStorage:
    """S3_BUCKET set -> S3-compatible (S3_ENDPOINT_URL for R2/Linode/MinIO); else local dir."""
    if os.environ.get("S3_BUCKET"):
        return S3Storage(os.environ["S3_BUCKET"], os.environ.get("S3_ENDPOINT_URL"),
                         os.environ.get("S3_REGION", "auto"),
                         os.environ.get("S3_ACCESS_KEY"), os.environ.get("S3_SECRET_KEY"),
                         public_endpoint_url=os.environ.get("S3_PUBLIC_ENDPOINT_URL"))
    return LocalStorage(os.environ.get("STORAGE_DIR", "./storage"))
