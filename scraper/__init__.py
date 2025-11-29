"""
Universal Web Scraper - Open Source Alternative to Firecrawl

A comprehensive web scraping and crawling system that can:
- Scrape any website (static or JavaScript-heavy)
- Extract and transcribe media content
- Generate RAG-ready data
- Support batch processing and scheduling
- Provide multiple output formats

Author: MiniMax Agent
License: MIT
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"

from .core.engine import ScrapingEngine
from .core.crawler import WebCrawler
from .extractors.content_extractor import ContentExtractor
from .extractors.media_extractor import MediaExtractor
from .processors.ocr_processor import OCRProcessor
from .processors.transcription_processor import TranscriptionProcessor
from .storage.vector_storage import VectorStorage
from .storage.chunking import TextChunker
from .api.server import app

__all__ = [
    "ScrapingEngine",
    "WebCrawler", 
    "ContentExtractor",
    "MediaExtractor",
    "OCRProcessor",
    "TranscriptionProcessor",
    "VectorStorage",
    "TextChunker",
    "app"
]