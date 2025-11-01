from pydantic import BaseModel
from typing import List


class SegmentMetadataNova(BaseModel):
    segmentIndex: int
    segmentStartSeconds: float
    segmentEndSeconds: float


class VideoEmbeddingSegmentNova(BaseModel):
    embedding: List[float]
    status: str
    segmentMetadata: SegmentMetadataNova

    def __getitem__(self, item):
        return getattr(self, item)


class VideoEmbeddingsNova(BaseModel):
    videoName: str
    s3URI: str
    keyframeURL: str
    dateCreated: str
    sizeBytes: int
    durationSec: float = 0.0
    contentType: str
    embeddings: List[VideoEmbeddingSegmentNova]

    def __getitem__(self, item):
        return getattr(self, item)


# Data model for TwelveLabs Pegasus model for video analysis structure
class VideoAnalysisPegasus(BaseModel):
    videoName: str
    s3URI: str
    title: str
    summary: str
    keywords: list[str] = []
    dateCreated: str


class OpenSearchDocumentNova(BaseModel):
    videoName: str
    s3URI: str
    keyframeURL: str
    title: str
    summary: str
    keywords: List[str]
    dateCreated: str
    contentType: str
    sizeBytes: int
    durationSec: float = 0.0
    embeddings: List[VideoEmbeddingSegmentNova]
