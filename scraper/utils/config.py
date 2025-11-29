"""
Configuration management for the Universal Web Scraper.

Provides comprehensive configuration classes for all components.
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScraperConfig:
    """Configuration for the main scraping engine."""
    
    # Browser settings
    headless: bool = True
    user_agent: str = "UniversalScraper/1.0 (compatible; MSIE 7.0; Windows NT 5.1)"
    timeout: int = 30
    viewport_width: int = 1920
    viewport_height: int = 1080
    
    # Proxy settings
    proxy: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    
    # Request settings
    max_concurrent_requests: int = 5
    delay_between_requests: float = 1.0
    requests_per_second: float = 1.0
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Headers
    custom_headers: Optional[Dict[str, str]] = None
    
    # Storage settings
    storage_enabled: bool = True
    storage_path: Path = field(default_factory=lambda: Path("scraped_data"))
    storage_format: str = "json"  # json, markdown, html
    
    # JavaScript and dynamic content
    wait_for_network_idle: bool = True
    wait_for_selector_timeout: int = 10000
    enable_javascript: bool = True
    
    # Anti-detection
    stealth_mode: bool = True
    random_user_agents: bool = True
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)


@dataclass
class CrawlerConfig:
    """Configuration for web crawling."""
    
    # Crawling limits
    max_depth: Optional[int] = 3
    max_pages: Optional[int] = 1000
    max_concurrent_requests: int = 3
    
    # URL filtering
    stay_on_domain: bool = True
    allowed_extensions: Optional[List[str]] = None
    excluded_extensions: Optional[List[str]] = None
    url_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    
    # Politeness
    respect_robots_txt: bool = True
    requests_per_second: float = 1.0
    delay_between_requests: float = 2.0
    timeout: int = 30
    user_agent: str = "UniversalScraperCrawler/1.0"
    
    # Scraping configuration
    scraping_config: ScraperConfig = field(default_factory=ScraperConfig)
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 1000000
    filter_duplicates: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                '.html', '.htm', '.php', '.asp', '.aspx', '.jsp',
                '.txt', '.md', '.pdf', '.doc', '.docx'
            ]
            
        if self.excluded_extensions is None:
            self.excluded_extensions = [
                '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg',
                '.mp4', '.avi', '.mov', '.mkv', '.webm',
                '.mp3', '.wav', '.flac', '.ogg',
                '.zip', '.rar', '.7z', '.tar', '.gz'
            ]


@dataclass
class ContentExtractorConfig:
    """Configuration for content extraction."""
    
    # Content selection
    content_selectors: List[str] = field(default_factory=lambda: [
        'main', 'article', '[role="main"]', '.content', '.post-content',
        '.entry-content', '.article-content', '.post-body', '.article-body',
        '#content', '#main-content', '.main-content'
    ])
    
    unwanted_selectors: List[str] = field(default_factory=lambda: [
        'nav', 'header', 'footer', 'aside', '.sidebar', '.navigation',
        '.menu', '.breadcrumb', '.social-share', '.comments', '.related-posts',
        '.advertisement', '.ads', '.banner', '.popup', '.modal',
        '[class*="advert"]', '[class*="sponsor"]', '[class*="promo"]'
    ])
    
    # Extraction options
    extract_images: bool = True
    extract_links: bool = True
    extract_metadata: bool = True
    extract_structured_data: bool = True
    extract_tables: bool = True
    extract_code_blocks: bool = True
    
    # Content processing
    clean_html: bool = True
    remove_scripts: bool = True
    remove_styles: bool = True
    convert_to_markdown: bool = True
    preserve_formatting: bool = True
    
    # Text processing
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    extract_text_only: bool = False


@dataclass
class MediaExtractorConfig:
    """Configuration for media extraction."""
    
    # Download settings
    download_enabled: bool = True
    download_path: Path = field(default_factory=lambda: Path("media_downloads"))
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_concurrent_downloads: int = 3
    
    # Image processing
    process_images: bool = True
    image_format: str = "jpg"
    image_quality: int = 90
    max_image_width: int = 1920
    max_image_height: int = 1080
    
    # Video processing
    process_videos: bool = True
    extract_audio_from_videos: bool = True
    video_format: str = "mp4"
    max_video_duration: int = 3600  # 1 hour
    extract_key_frames: bool = True
    key_frame_count: int = 5
    
    # Audio processing
    process_audio: bool = True
    audio_format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1  # Mono
    
    # Supported formats
    supported_image_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff'
    ])
    supported_video_formats: List[str] = field(default_factory=lambda: [
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'
    ])
    supported_audio_formats: List[str] = field(default_factory=lambda: [
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'
    ])


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    
    # Tesseract settings
    languages: List[str] = field(default_factory=lambda: ['eng'])
    engine: str = "tesseract"  # tesseract, opencv_tesseract
    oem: int = 3  # OCR Engine Mode
    psm: int = 6  # Page Segmentation Mode
    
    # Preprocessing
    enable_preprocessing: bool = True
    contrast_enhancement: float = 1.2
    sharpness_enhancement: float = 1.2
    denoising: bool = True
    
    # Output options
    include_word_positions: bool = True
    include_confidence_scores: bool = True
    output_format: str = "text"  # text, hocr, tsv
    
    # Performance
    max_concurrent: int = 2
    batch_processing: bool = True
    
    # Language detection
    auto_detect_language: bool = True
    confidence_threshold: float = 30.0


@dataclass
class TranscriptionConfig:
    """Configuration for speech transcription."""
    
    # Whisper settings
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # None for auto-detect
    task: str = "transcribe"  # transcribe, translate
    
    # Audio processing
    sample_rate: int = 16000
    channels: int = 1  # Mono
    normalize_audio: bool = True
    
    # Performance
    max_concurrent: int = 1
    batch_processing: bool = True
    
    # Output options
    include_timestamps: bool = True
    include_confidence: bool = True
    word_level_timestamps: bool = False
    
    # Preprocessing
    enable_preprocessing: bool = True
    noise_reduction: bool = True
    silence_removal: bool = True
    max_segment_duration: float = 30.0  # seconds
    
    # Speaker diarization
    enable_speaker_diarization: bool = False
    min_speakers: int = 1
    max_speakers: int = 10


@dataclass
class VectorStorageConfig:
    """Configuration for vector storage."""
    
    # Storage backend
    backend: str = "chromadb"  # chromadb, faiss, qdrant
    
    # Collection settings
    collection_name: str = "universal_scraper"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: Optional[int] = None
    
    # Storage paths
    storage_path: Path = field(default_factory=lambda: Path("vector_storage"))
    
    # FAISS specific
    index_type: str = "Flat"  # Flat, HNSW, IVF
    nlist: int = 100  # For IVF indexes
    
    # Qdrant specific
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    
    # Performance
    batch_size: int = 100
    max_concurrent: int = 3
    
    # Search settings
    default_search_limit: int = 10
    similarity_threshold: float = 0.7


@dataclass
class TextChunkingConfig:
    """Configuration for text chunking."""
    
    # Chunking strategy
    strategy: str = "semantic"  # fixed_size, sentence, semantic, markdown, recursive
    
    # Chunk size settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 4000
    
    # Sentence preservation
    preserve_sentences: bool = True
    sentence_boundary_tolerance: int = 50
    
    # Semantic chunking
    similarity_threshold: float = 0.7
    semantic_window_size: int = 3
    
    # Markdown awareness
    respect_headings: bool = True
    respect_code_blocks: bool = True
    respect_lists: bool = True
    
    # Language settings
    language: str = "en"
    
    # Performance
    max_chunks_per_document: int = 1000
    
    # Output formatting
    include_metadata: bool = True
    include_positions: bool = True
    include_embeddings: bool = False  # Set to True for semantic chunking


@dataclass
class APIConfig:
    """Configuration for the API server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Authentication
    auth_enabled: bool = False
    auth_secret: Optional[str] = None
    auth_token_expiry: int = 3600  # seconds
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 100
    
    # Job management
    max_concurrent_jobs: int = 5
    job_timeout: int = 3600  # seconds
    job_cleanup_interval: int = 300  # seconds
    
    # Storage
    database_url: Optional[str] = None
    redis_url: Optional[str] = None


@dataclass
class DatabaseConfig:
    """Configuration for database storage."""
    
    # Database type
    type: str = "sqlite"  # sqlite, postgresql, mysql
    
    # SQLite settings
    sqlite_path: Path = field(default_factory=lambda: Path("scraper.db"))
    
    # PostgreSQL settings
    postgresql_url: Optional[str] = None
    postgresql_host: str = "localhost"
    postgresql_port: int = 5432
    postgresql_database: str = "scraper"
    postgresql_username: str = "scraper"
    postgresql_password: Optional[str] = None
    
    # Connection settings
    connection_pool_size: int = 10
    connection_timeout: int = 30
    max_retries: int = 3
    
    # Migration settings
    auto_migrate: bool = True
    migration_path: Path = field(default_factory=lambda: Path("migrations"))


class Config:
    """Main configuration class that combines all component configs."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.scraper = ScraperConfig()
        self.crawler = CrawlerConfig()
        self.content_extractor = ContentExtractorConfig()
        self.media_extractor = MediaExtractorConfig()
        self.ocr = OCRConfig()
        self.transcription = TranscriptionConfig()
        self.vector_storage = VectorStorageConfig()
        self.chunking = TextChunkingConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        
        # Load configuration from file if provided
        if config_file:
            self.load_from_file(config_file)
            
        # Load from environment variables
        self.load_from_env()
        
    def load_from_file(self, config_file: Union[str, Path]):
        """Load configuration from file."""
        import json
        
        config_path = Path(config_file)
        if not config_path.exists():
            return
            
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Update configurations based on file data
            # This is a simplified implementation
            # In practice, you'd want more sophisticated parsing
            
        except Exception as e:
            print(f"Failed to load config file: {e}")
            
    def load_from_env(self):
        """Load configuration from environment variables."""
        
        # API configuration
        if os.getenv("SCRAPER_API_HOST"):
            self.api.host = os.getenv("SCRAPER_API_HOST")
        if os.getenv("SCRAPER_API_PORT"):
            self.api.port = int(os.getenv("SCRAPER_API_PORT"))
        if os.getenv("SCRAPER_API_DEBUG"):
            self.api.debug = os.getenv("SCRAPER_API_DEBUG").lower() == "true"
            
        # Database configuration
        if os.getenv("DATABASE_URL"):
            self.database.type = "postgresql"
            self.database.postgresql_url = os.getenv("DATABASE_URL")
            
        # Vector storage configuration
        if os.getenv("VECTOR_STORAGE_BACKEND"):
            self.vector_storage.backend = os.getenv("VECTOR_STORAGE_BACKEND")
        if os.getenv("QDRANT_URL"):
            self.vector_storage.qdrant_url = os.getenv("QDRANT_URL")
            
        # Scraping configuration
        if os.getenv("SCRAPER_HEADLESS"):
            self.scraper.headless = os.getenv("SCRAPER_HEADLESS").lower() == "true"
        if os.getenv("SCRAPER_PROXY"):
            self.scraper.proxy = os.getenv("SCRAPER_PROXY")
        if os.getenv("SCRAPER_USER_AGENT"):
            self.scraper.user_agent = os.getenv("SCRAPER_USER_AGENT")
            
        # Transcription configuration
        if os.getenv("WHISPER_MODEL_SIZE"):
            self.transcription.model_size = os.getenv("WHISPER_MODEL_SIZE")
            
    def save_to_file(self, config_file: Union[str, Path]):
        """Save current configuration to file."""
        import json
        
        config_data = {
            "scraper": self.scraper.__dict__,
            "crawler": self.crawler.__dict__,
            "content_extractor": self.content_extractor.__dict__,
            "media_extractor": self.media_extractor.__dict__,
            "ocr": self.ocr.__dict__,
            "transcription": self.transcription.__dict__,
            "vector_storage": self.vector_storage.__dict__,
            "chunking": self.chunking.__dict__,
            "api": self.api.__dict__,
            "database": self.database.__dict__
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
            
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate scraping configuration
        if self.scraper.max_concurrent_requests < 1:
            issues.append("scraper.max_concurrent_requests must be at least 1")
            
        if self.scraper.requests_per_second <= 0:
            issues.append("scraper.requests_per_second must be positive")
            
        # Validate crawler configuration
        if self.crawler.max_depth and self.crawler.max_depth < 0:
            issues.append("crawler.max_depth cannot be negative")
            
        if self.crawler.max_pages and self.crawler.max_pages < 1:
            issues.append("crawler.max_pages must be at least 1")
            
        # Validate chunking configuration
        if self.chunking.chunk_size < self.chunking.chunk_overlap:
            issues.append("chunking.chunk_size must be greater than chunk_overlap")
            
        if self.chunking.chunk_overlap < 0:
            issues.append("chunking.chunk_overlap cannot be negative")
            
        # Validate API configuration
        if self.api.port < 1 or self.api.port > 65535:
            issues.append("api.port must be between 1 and 65535")
            
        return issues
        
    def get_database_url(self) -> str:
        """Get database URL based on configuration."""
        if self.database.type == "sqlite":
            return f"sqlite:///{self.database.sqlite_path}"
        elif self.database.type == "postgresql":
            if self.database.postgresql_url:
                return self.database.postgresql_url
            else:
                username = self.database.postgresql_username
                password = self.database.postgresql_password
                auth = f"{username}:{password}@" if password else f"{username}@"
                return f"postgresql://{auth}{self.database.postgresql_host}:{self.database.postgresql_port}/{self.database.postgresql_database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")