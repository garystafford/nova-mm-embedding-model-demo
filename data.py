from pydantic import BaseModel
from typing import List


class SegmentMetadata(BaseModel):
    segmentIndex: int
    segmentStartSeconds: float
    segmentEndSeconds: float


class VideoEmbeddingSegment(BaseModel):
    embedding: List[float]
    status: str
    segmentMetadata: SegmentMetadata


class VideoEmbeddings(BaseModel):
    videoName: str
    s3URI: str
    keyframeURL: str
    dateCreated: str
    sizeBytes: int
    durationSec: float = 0.0
    contentType: str
    embeddings: List[VideoEmbeddingSegment]


class VideoAnalysis(BaseModel):
    videoName: str
    s3URI: str
    title: str
    summary: str
    keywords: list[str] = []
    dateCreated: str


class OpenSearchDocument(BaseModel):
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
    embeddings: List[VideoEmbeddingSegment]
