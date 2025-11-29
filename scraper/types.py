"""
Common types and data structures for the Universal Web Scraper.
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json


@dataclass
class MediaContent:
    """Represents extracted media content."""
    
    type: str  # 'image', 'video', 'audio'
    url: str
    filename: Optional[str] = None
    size: Optional[int] = None
    mime_type: Optional[str] = None
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    duration: Optional[float] = None  # For audio/video
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ExtractedLink:
    """Represents a discovered link."""
    
    url: str
    text: Optional[str] = None
    title: Optional[str] = None
    rel: Optional[str] = None
    target: Optional[str] = None


@dataclass
class ScrapedPage:
    """Represents a scraped web page."""
    
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    html_content: Optional[str] = None
    markdown_content: Optional[str] = None
    
    # Metadata
    meta_description: Optional[str] = None
    meta_keywords: Optional[str] = None
    meta_author: Optional[str] = None
    meta_language: Optional[str] = None
    canonical_url: Optional[str] = None
    
    # Content analysis
    word_count: Optional[int] = None
    reading_time: Optional[int] = None  # seconds
    language_detected: Optional[str] = None
    
    # Media and links
    images: List[MediaContent] = field(default_factory=list)
    videos: List[MediaContent] = field(default_factory=list)
    audio: List[MediaContent] = field(default_factory=list)
    links: List[ExtractedLink] = field(default_factory=list)
    
    # Timestamps
    scraped_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    
    # Performance metrics
    load_time: Optional[float] = None  # seconds
    request_count: Optional[int] = None
    total_size: Optional[int] = None  # bytes
    
    # Processing status
    status: str = "success"  # success, error, partial
    error_message: Optional[str] = None
    processing_time: Optional[float] = None  # seconds


@dataclass
class ScrapingResult:
    """Complete result of a scraping operation."""
    
    # Summary
    total_pages: int
    successful_pages: int
    failed_pages: int
    total_urls_found: int
    
    # Content
    pages: List[ScrapedPage] = field(default_factory=list)
    discovered_urls: List[str] = field(default_factory=list)
    
    # Statistics
    total_content_length: int = 0
    total_images: int = 0
    total_videos: int = 0
    total_audio: int = 0
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None
    
    # Configuration used
    config_snapshot: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_pages": self.total_pages,
                "successful_pages": self.successful_pages,
                "failed_pages": self.failed_pages,
                "total_urls_found": self.total_urls_found,
                "total_content_length": self.total_content_length,
                "total_images": self.total_images,
                "total_videos": self.total_videos,
                "total_audio": self.total_audio,
            },
            "metadata": {
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "total_duration": self.total_duration,
            },
            "pages": [page.__dict__ for page in self.pages],
            "discovered_urls": self.discovered_urls,
            "config_snapshot": self.config_snapshot,
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_to_file(self, filepath: Path) -> None:
        """Save result to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


@dataclass
class CrawlJob:
    """Represents a crawling job."""
    
    start_urls: List[str]
    max_depth: int = 0
    max_pages: Optional[int] = None
    follow_external_links: bool = False
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    
    # Filtering
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    require_patterns: Optional[List[str]] = None
    
    # Job metadata
    job_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, running, completed, failed, stopped
    progress: float = 0.0
    
    # Results
    result: Optional[ScrapingResult] = None
    error_message: Optional[str] = None


# Content extraction types
ContentExtractionResult = Dict[str, Any]
MediaExtractionResult = Dict[str, List[MediaContent]]


# Vector storage types
VectorDocument = Dict[str, Any]
VectorSearchResult = Dict[str, Union[str, float]]


# API types
ScrapeRequest = Dict[str, Any]
ScrapeResponse = Dict[str, Any]
CrawlRequest = Dict[str, Any]
CrawlResponse = Dict[str, Any]