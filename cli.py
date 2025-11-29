#!/usr/bin/env python3
"""
Universal Web Scraper CLI

Command-line interface for the Universal Web Scraper.
Provides easy access to all scraping and crawling features.
"""

import asyncio
import click
import json
from pathlib import Path
from datetime import datetime
from typing import List

from scraper import (
    ScrapingEngine,
    WebCrawler,
    MediaExtractor,
    VectorStorage,
    TextChunker
)
from scraper.utils.config import ScraperConfig
from scraper.utils.logger import setup_logging


@click.group()
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Universal Web Scraper - Open-source web scraping and crawling tool."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level, structured_logging=False)
    
    # Load configuration
    if config:
        ctx.obj['config'] = Config(config)
    else:
        ctx.obj['config'] = Config()


@cli.command()
@click.argument('url')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['json', 'markdown', 'html']), 
              default='markdown', help='Output format')
@click.option('--screenshot/--no-screenshot', default=False, help='Take screenshot')
@click.option('--store-media/--no-store-media', default=True, help='Store media files')
@click.pass_context
def scrape(ctx, url, output, output_format, screenshot, store_media):
    """Scrape a single URL."""
    config = ctx.obj['config']
    
    async def _scrape():
        options = {
            'convert_to_markdown': output_format == 'markdown',
            'screenshot': screenshot,
            'store_media': store_media
        }
        
        async with ScrapingEngine(config.scraper) as engine:
            click.echo(f"Scraping {url}...")
            result = await engine.scrape(url, options)
            
            if result.success:
                click.echo(f"✅ Success! Title: {result.title}")
                click.echo(f"Processing time: {result.processing_time:.2f}s")
                
                # Save output
                if output:
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_format == 'json':
                        data = {
                            'url': result.url,
                            'title': result.title,
                            'content': result.content,
                            'markdown': result.markdown,
                            'metadata': result.metadata,
                            'timestamp': result.timestamp.isoformat(),
                            'processing_time': result.processing_time
                        }
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    elif output_format == 'markdown':
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {result.title}\n\n")
                            f.write(f"**Source:** {result.url}\n")
                            f.write(f"**Scraped:** {result.timestamp.isoformat()}\n\n")
                            f.write("---\n\n")
                            f.write(result.markdown or result.content)
                    elif output_format == 'html':
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(result.html)
                    
                    click.echo(f"Output saved to {output}")
                else:
                    # Print to stdout
                    if output_format == 'markdown' and result.markdown:
                        click.echo(result.markdown)
                    elif result.content:
                        click.echo(result.content)
                        
            else:
                click.echo(f"❌ Failed: {result.error_message}", err=True)
                
    asyncio.run(_scrape())


@cli.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--output-dir', '-d', type=click.Path(), default='./scraped', 
              help='Output directory for results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'markdown']), 
              default='json', help='Output format')
@click.option('--max-concurrent', '-c', type=int, default=3, 
              help='Maximum concurrent requests')
@click.pass_context
def scrape_batch(ctx, urls, output_dir, output_format, max_concurrent):
    """Scrape multiple URLs in batch."""
    config = ctx.obj['config']
    config.scraper.max_concurrent_requests = max_concurrent
    
    async def _batch_scrape():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        async with ScrapingEngine(config.scraper) as engine:
            click.echo(f"Batch scraping {len(urls)} URLs...")
            
            options = {
                'convert_to_markdown': output_format == 'markdown',
                'screenshot': False,
                'store_media': False
            }
            
            results = await engine.scrape_batch(list(urls), options)
            
            successful = sum(1 for r in results if r.success)
            click.echo(f"✅ Completed: {successful}/{len(results)} successful")
            
            # Save results
            for i, result in enumerate(results):
                if result.success:
                    filename = f"{i+1:03d}_{result.title[:50].replace('/', '_')}.{output_format}"
                    filepath = output_path / filename
                    
                    if output_format == 'json':
                        data = {
                            'url': result.url,
                            'title': result.title,
                            'content': result.content,
                            'markdown': result.markdown,
                            'timestamp': result.timestamp.isoformat(),
                            'processing_time': result.processing_time
                        }
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    elif output_format == 'markdown':
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"# {result.title}\n\n")
                            f.write(f"**Source:** {result.url}\n")
                            f.write(f"**Scraped:** {result.timestamp.isoformat()}\n\n")
                            f.write("---\n\n")
                            f.write(result.markdown or result.content)
                            
            click.echo(f"Results saved to {output_path}")
                
    asyncio.run(_batch_scrape())


@cli.command()
@click.argument('url')
@click.option('--max-depth', '-d', type=int, default=3, help='Maximum crawl depth')
@click.option('--max-pages', '-p', type=int, default=100, help='Maximum pages to crawl')
@click.option('--stay-domain/--allow-subdomains', default=True, 
              help='Stay on same domain')
@click.option('--respect-robots/--ignore-robots', default=True, 
              help='Respect robots.txt')
@click.option('--output', '-o', type=click.Path(), help='Output file for crawl results')
@click.pass_context
def crawl(ctx, url, max_depth, max_pages, stay_domain, respect_robots, output):
    """Crawl a website."""
    config = ctx.obj['config']
    config.crawler.max_depth = max_depth
    config.crawler.max_pages = max_pages
    config.crawler.stay_on_domain = stay_domain
    config.crawler.respect_robots_txt = respect_robots
    
    async def _crawl():
        async with WebCrawler(config.crawler) as crawler:
            click.echo(f"Starting crawl of {url}")
            click.echo(f"Max depth: {max_depth}, Max pages: {max_pages}")
            
            job = await crawler.crawl_website(
                start_url=url,
                max_depth=max_depth,
                max_pages=max_pages,
                stay_on_domain=stay_domain,
                respect_robots_txt=respect_robots
            )
            
            click.echo(f"Job ID: {job.id}")
            
            # Monitor progress
            with click.progressbar(length=100, label='Crawling') as bar:
                while job.status in ['pending', 'running']:
                    await asyncio.sleep(2)
                    job = crawler.get_job(job.id)
                    progress = int(job.progress * 100)
                    bar.update(progress - bar.pos)
            
            click.echo(f"\n✅ Crawl completed!")
            click.echo(f"Status: {job.status}")
            click.echo(f"Discovered: {job.urls_discovered} URLs")
            click.echo(f"Crawled: {job.urls_crawled} pages")
            click.echo(f"Successful: {job.urls_successful} pages")
            click.echo(f"Failed: {job.urls_failed} pages")
            
            # Save results
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                results_data = {
                    'job_id': job.id,
                    'start_url': job.start_url,
                    'status': job.status,
                    'stats': {
                        'urls_discovered': job.urls_discovered,
                        'urls_crawled': job.urls_crawled,
                        'urls_successful': job.urls_successful,
                        'urls_failed': job.urls_failed
                    },
                    'results': [
                        {
                            'url': result.url,
                            'title': result.title,
                            'content': result.content[:500],
                            'success': result.success,
                            'processing_time': result.processing_time
                        }
                        for result in job.results
                    ]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                
                click.echo(f"Results saved to {output_path}")
                
    asyncio.run(_crawl())


@cli.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--output-dir', '-d', type=click.Path(), default='./media', 
              help='Output directory for media files')
@click.option('--process-images/--skip-images', default=True, 
              help='Process images with OCR')
@click.option('--process-videos/--skip-videos', default=True, 
              help='Process videos (extract audio and transcribe)')
@click.option('--process-audio/--skip-audio', default=True, 
              help='Process audio files (transcribe)')
@click.pass_context
def extract_media(ctx, urls, output_dir, process_images, process_videos, process_audio):
    """Extract and process media from URLs."""
    
    async def _extract_media():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        async with MediaExtractor(download_dir=output_path) as media_extractor:
            click.echo(f"Extracting media from {len(urls)} URLs...")
            
            results = await media_extractor.extract_media_from_page(
                media_urls=list(urls),
                process_images=process_images,
                process_videos=process_videos,
                process_audio=process_audio
            )
            
            # Summary
            click.echo(f"\n📊 Extraction Summary:")
            click.echo(f"Images: {len(results['images'])}")
            click.echo(f"Videos: {len(results['videos'])}")
            click.echo(f"Audio: {len(results['audio'])}")
            click.echo(f"Documents: {len(results['documents'])}")
            
            # Show some results
            if results['images']:
                click.echo(f"\n🖼️  Image processing results:")
                for img in results['images'][:3]:  # Show first 3
                    if img.get('ocr_text'):
                        click.echo(f"  - {Path(img['local_path']).name}: OCR text found")
                    else:
                        click.echo(f"  - {Path(img['local_path']).name}: No text detected")
                        
            if results['audio']:
                click.echo(f"\n🎵 Audio transcription results:")
                for audio in results['audio'][:3]:  # Show first 3
                    if audio.get('transcription'):
                        text_preview = audio['transcription'][:100] + "..."
                        click.echo(f"  - {Path(audio['local_path']).name}: {text_preview}")
                    else:
                        click.echo(f"  - {Path(audio['local_path']).name}: No transcription")
            
            click.echo(f"\nFiles saved to: {output_path}")
                
    asyncio.run(_extract_media())


@cli.command()
@click.argument('text_file', type=click.Path(exists=True))
@click.option('--output-dir', '-d', type=click.Path(), default='./chunks', 
              help='Output directory for chunks')
@click.option('--chunk-size', type=int, default=1000, help='Chunk size in characters')
@click.option('--chunk-overlap', type=int, default=200, help='Chunk overlap in characters')
@click.option('--strategy', type=click.Choice(['fixed_size', 'sentence', 'semantic', 'markdown']), 
              default='semantic', help='Chunking strategy')
@click.option('--backend', type=click.Choice(['chromadb', 'faiss', 'qdrant']), 
              default='chromadb', help='Vector database backend')
@click.pass_context
def create_rag_index(ctx, text_file, output_dir, chunk_size, chunk_overlap, strategy, backend):
    """Create RAG-ready chunks and vector index from text."""
    
    async def _create_index():
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        click.echo(f"Creating RAG index from {text_file}")
        click.echo(f"Text length: {len(text)} characters")
        
        # Create chunks
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=strategy
        )
        
        chunks = chunker.chunk_text(
            text=text,
            document_id=Path(text_file).stem,
            metadata={'source_file': str(text_file)}
        )
        
        click.echo(f"Created {len(chunks)} chunks")
        
        # Create vector index
        config = ctx.obj['config']
        config.vector_storage.backend = backend
        
        async with VectorStorage(config.vector_storage) as vector_db:
            # Add chunks to vector database
            chunk_docs = [
                {
                    'text': chunk.text,
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            ]
            
            doc_ids = await vector_db.add_documents(chunk_docs)
            click.echo(f"Indexed {len(doc_ids)} chunks in {backend}")
            
            # Save chunks to file
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            chunks_file = output_path / f"{Path(text_file).stem}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump([
                    {
                        'id': chunk.id,
                        'text': chunk.text,
                        'metadata': chunk.metadata
                    }
                    for chunk in chunks
                ], f, indent=2, ensure_ascii=False)
            
            click.echo(f"Chunks saved to {chunks_file}")
            
            # Test search
            click.echo("\n🔍 Testing search...")
            test_results = await vector_db.search(
                query="test search query",
                limit=3
            )
            
            click.echo(f"Found {len(test_results)} relevant chunks")
            for i, result in enumerate(test_results):
                preview = result['text'][:80] + "..."
                click.echo(f"{i+1}. {preview} (score: {result['score']:.3f})")
                
    asyncio.run(_create_index())


@cli.command()
@click.option('--backend', type=click.Choice(['chromadb', 'faiss', 'qdrant']), 
              default='chromadb', help='Vector database backend')
@click.option('--test-query', type=str, help='Test search query')
@click.pass_context
def test_vector_db(ctx, backend, test_query):
    """Test vector database functionality."""
    
    async def _test():
        config = ctx.obj['config']
        config.vector_storage.backend = backend
        
        async with VectorStorage(config.vector_storage) as vector_db:
            click.echo(f"Testing {backend} vector database...")
            
            # Test documents
            test_docs = [
                {
                    'text': 'Python is a programming language created by Guido van Rossum.',
                    'metadata': {'topic': 'programming', 'language': 'python'}
                },
                {
                    'text': 'Machine learning is a subset of artificial intelligence.',
                    'metadata': {'topic': 'ai', 'category': 'machine_learning'}
                },
                {
                    'text': 'Web scraping involves extracting data from websites.',
                    'metadata': {'topic': 'web', 'category': 'data_extraction'}
                }
            ]
            
            # Add documents
            doc_ids = await vector_db.add_documents(test_docs)
            click.echo(f"Added {len(doc_ids)} test documents")
            
            # Test search
            if test_query:
                results = await vector_db.search(query=test_query, limit=5)
                click.echo(f"\nSearch results for '{test_query}':")
                for i, result in enumerate(results):
                    click.echo(f"{i+1}. {result['text']} (score: {result['score']:.3f})")
            else:
                click.echo("No test query provided, performing sample searches...")
                
                # Default queries
                queries = ['programming language', 'artificial intelligence', 'data extraction']
                for query in queries:
                    results = await vector_db.search(query=query, limit=3)
                    click.echo(f"\nQuery: '{query}' - Found {len(results)} results")
                    for result in results:
                        click.echo(f"  - {result['text']} (score: {result['score']:.3f})")
            
            # Get stats
            stats = await vector_db.get_collection_stats()
            click.echo(f"\n📊 Database stats: {stats}")
                
    asyncio.run(_test())


@cli.command()
@click.pass_context
def status(ctx):
    """Show scraper status and system information."""
    click.echo("🕷️  Universal Web Scraper Status")
    click.echo("=" * 40)
    
    # System info
    import sys
    click.echo(f"Python version: {sys.version}")
    
    # Check core dependencies
    dependencies = {
        'playwright': 'Playwright',
        'fastapi': 'FastAPI',
        'spacy': 'spaCy',
        'sentence_transformers': 'Sentence Transformers',
        'chromadb': 'ChromaDB',
        'tesseract': 'Tesseract'
    }
    
    click.echo("\n📦 Dependencies:")
    for package, name in dependencies.items():
        try:
            __import__(package)
            click.echo(f"  ✅ {name}")
        except ImportError:
            click.echo(f"  ❌ {name} (not installed)")
    
    # Check external tools
    click.echo("\n🔧 External Tools:")
    import subprocess
    
    tools = {
        'tesseract --version': 'Tesseract OCR',
        'ffmpeg -version': 'FFmpeg (media processing)',
        'ffprobe -version': 'FFprobe (metadata extraction)'
    }
    
    for cmd, name in tools.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            click.echo(f"  ✅ {name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.echo(f"  ⚠️  {name} (not found - some features may not work)")
    
    click.echo("\n🌐 API Server: python -m scraper.api.server")
    click.echo("📖 Documentation: See README.md")


if __name__ == '__main__':
    cli()