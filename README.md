# Universal Web Scraper

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

A comprehensive, open-source web scraping and crawling framework written in Python. Built as a powerful alternative to commercial solutions like Firecrawl, with full support for JavaScript-rendered content, media extraction, OCR, transcription, and RAG-ready data storage.

## ✨ Features

### 🔥 Core Capabilities
- **Universal Web Scraping** - Scrape any website with robust error handling
- **Smart Website Crawling** - Systematically crawl entire sites with configurable depth
- **JavaScript Rendering** - Full support for modern, dynamic websites using Playwright
- **Media Extraction** - Download and process images, videos, and audio files
- **OCR Text Extraction** - Extract text from images using Tesseract
- **Audio/Video Transcription** - Convert speech to text using Whisper
- **Vector Storage** - RAG-ready storage with multiple backend support

### 🎯 Advanced Features
- **Anti-Detection** - Stealth browsing with rotating user agents
- **Rate Limiting** - Intelligent request throttling with domain-specific limits
- **Robots.txt Respect** - Built-in compliance with web crawling standards
- **Multiple Output Formats** - Markdown, JSON, HTML, and structured data
- **Batch Processing** - Efficient concurrent scraping of multiple URLs
- **Progress Tracking** - Real-time monitoring of crawl jobs
- **Extensible Architecture** - Plugin system for custom extractors

### 🛠 Technical Stack
- **Browser Automation**: Playwright for JavaScript support
- **Async Processing**: asyncio for high-performance concurrent operations
- **NLP Processing**: spaCy, NLTK, Transformers for text analysis
- **Machine Learning**: Sentence Transformers for semantic analysis
- **Vector Databases**: ChromaDB, FAISS, Qdrant support
- **OCR**: Tesseract for optical character recognition
- **Speech Recognition**: OpenAI Whisper for transcription
- **Web Framework**: FastAPI for REST API

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/universal-web-scraper.git
cd universal-web-scraper

# Install dependencies
pip install -r requirements.txt

# Download required models and data
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```

### Basic Usage

```python
import asyncio
from scraper import ScrapingEngine, Config

async def scrape_example():
    config = Config()
    
    async with ScrapingEngine(config.scraper) as engine:
        # Scrape a single page
        result = await engine.scrape("https://example.com")
        
        print(f"Title: {result.title}")
        print(f"Content: {result.markdown}")
        print(f"Success: {result.success}")

# Run the example
asyncio.run(scrape_example())
```

### Website Crawling

```python
from scraper import WebCrawler, CrawlerConfig

async def crawl_example():
    config = CrawlerConfig()
    
    async with WebCrawler(config) as crawler:
        # Start crawling a website
        job = await crawler.crawl_website(
            start_url="https://example.com",
            max_depth=3,
            max_pages=100
        )
        
        # Monitor progress
        while job.status == "running":
            await asyncio.sleep(5)
            job = crawler.get_job(job.id)
            print(f"Progress: {job.progress:.1%}")

asyncio.run(crawl_example())
```

### Media Processing

```python
from scraper import MediaExtractor, OCRProcessor, TranscriptionProcessor

async def media_example():
    async with MediaExtractor() as media_extractor:
        # Extract and process media
        results = await media_extractor.extract_media_from_page(
            media_urls=["image1.jpg", "video1.mp4", "audio1.mp3"],
            process_images=True,
            process_videos=True,
            process_audio=True
        )
        
        print(f"Images: {len(results['images'])}")
        print(f"Videos: {len(results['videos'])}")
        print(f"Audio: {len(results['audio'])}")

asyncio.run(media_example())
```

## 🏗 Architecture

```
scraper/
├── core/                   # Core scraping and crawling engines
│   ├── engine.py          # Main scraping engine
│   └── crawler.py         # Website crawler
├── extractors/            # Content extraction modules
│   ├── content_extractor.py
│   └── media_extractor.py
├── processors/            # Content processing
│   ├── ocr_processor.py
│   └── transcription_processor.py
├── storage/               # Storage and vector databases
│   ├── vector_storage.py
│   └── chunking.py
├── utils/                 # Utility modules
│   ├── config.py
│   ├── logger.py
│   ├── validators.py
│   ├── rate_limiter.py
│   └── storage.py
└── api/                   # FastAPI REST interface
    └── server.py
```

## 📖 Documentation

### Configuration

The scraper uses a comprehensive configuration system:

```python
from scraper.utils.config import Config

config = Config()

# Customize scraping behavior
config.scraper.headless = True
config.scraper.max_concurrent_requests = 5
config.crawler.max_depth = 3
config.vector_storage.backend = "chromadb"
```

### API Usage

Start the REST API server:

```bash
python -m scraper.api.server
```

API Endpoints:
- `POST /scrape` - Scrape a single URL
- `POST /scrape/batch` - Batch scrape multiple URLs
- `POST /crawl` - Start website crawling
- `GET /crawl/{job_id}/status` - Get crawl job status
- `GET /health` - Health check
- `GET /stats` - API statistics

### Content Processing

#### Text Chunking for RAG

```python
from scraper.storage.chunking import TextChunker

chunker = TextChunker(
    chunk_size=1000,
    chunk_overlap=200,
    strategy="semantic"  # fixed_size, sentence, semantic, markdown
)

chunks = chunker.chunk_text(
    text=your_text,
    document_id="doc_1",
    metadata={"source": "website"}
)
```

#### Vector Storage

```python
from scraper.storage.vector_storage import VectorStorage
from scraper.utils.config import VectorStorageConfig

config = VectorStorageConfig(backend="chromadb")

async with VectorStorage(config) as vector_db:
    # Add documents
    doc_ids = await vector_db.add_documents([
        {"text": "Your document text", "metadata": {"source": "url"}}
    ])
    
    # Search for similar content
    results = await vector_db.search(
        query="search query",
        limit=10
    )
```

## 🧪 Examples

Check the `examples/` directory for comprehensive usage examples:

- `basic_usage.py` - Basic scraping and crawling examples
- `media_processing.py` - Media extraction and processing
- `rag_pipeline.py` - Complete RAG pipeline with vector storage
- `api_client.py` - REST API client examples

## 🎯 Use Cases

### 1. **Data Collection**
- Extract structured data from websites
- Monitor competitor pricing and products
- Collect news articles and blog posts
- Gather research data from academic sites

### 2. **Content Analysis**
- Analyze website content for SEO
- Extract metadata and schema.org data
- Process multimedia content with OCR/transcription
- Generate content summaries and insights

### 3. **RAG Applications**
- Build knowledge bases from web content
- Create searchable document collections
- Generate embeddings for semantic search
- Feed data to LLM applications

### 4. **Monitoring & Alerts**
- Track changes on websites
- Monitor for new content
- Alert on specific triggers
- Automated content updates

## 🔧 Advanced Features

### Rate Limiting
```python
from scraper.utils.rate_limiter import DomainRateLimiter

limiter = DomainRateLimiter(
    default_requests_per_second=1.0,
    default_max_burst=3
)

# Domain-specific limits
limiter.set_domain_rate("example.com", requests_per_second=0.5)
```

### Custom Extractors
```python
class CustomExtractor:
    async def extract(self, soup):
        # Your custom extraction logic
        return extracted_data

# Register with the engine
engine.register_extractor("custom", CustomExtractor())
```

### Media Processing
```python
# OCR on images
async with OCRProcessor() as ocr:
    result = await ocr.extract_text("image.png")
    print(result['text'])

# Transcribe audio/video
async with TranscriptionProcessor() as transcriber:
    result = await transcriber.transcribe_audio("audio.mp3")
    print(result['text'])
```

## 🚨 Important Notes

### Respectful Crawling
- Always respect robots.txt
- Use appropriate rate limiting
- Don't overwhelm target servers
- Check terms of service

### Legal Considerations
- Only scrape publicly available content
- Respect copyright and intellectual property
- Check local laws and regulations
- Be ethical in your scraping practices

### Performance Considerations
- Adjust concurrent requests based on target site
- Monitor memory usage for large crawls
- Use appropriate chunk sizes for RAG
- Clean up temporary files regularly


### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-username/universal-web-scraper.git
cd universal-web-scraper

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy scraper/

# Format code
black scraper/
isort scraper/
```

## 📊 Performance

Benchmarks against similar tools:

| Feature | Universal Scraper | Firecrawl | Scrapy |
|---------|------------------|-----------|--------|
| JavaScript Support | ✅ Excellent | ✅ Excellent | ❌ Limited |
| Batch Processing | ✅ Native | ✅ Native | ✅ Manual |
| Media Processing | ✅ Full Stack | ❌ Limited | ❌ None |
| Vector Storage | ✅ Multiple Backends | ❌ None | ❌ None |
| Rate Limiting | ✅ Advanced | ✅ Basic | ⚠️ Manual |
| API Server | ✅ FastAPI | ✅ REST API | ❌ None |
| Open Source | ✅ MIT | ⚠️ Mixed | ✅ MIT |

## 🛡️ Security

- All dependencies are regularly updated
- Input validation and sanitization
- Secure storage of scraped data
- Rate limiting prevents abuse
- No execution of remote code

## 🙏 Acknowledgments

- [Playwright](https://playwright.dev/) for browser automation
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR
- [Whisper](https://github.com/openai/whisper) for speech recognition
- [spaCy](https://spacy.io/) for NLP processing
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the web API

