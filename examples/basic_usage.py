"""
Example usage of the Universal Web Scraper

This script demonstrates how to use the Universal Web Scraper
to scrape websites, crawl sites, and process media content.
"""

import asyncio
import json
from pathlib import Path

from scraper import (
    ScrapingEngine,
    WebCrawler,
    MediaExtractor,
    OCRProcessor,
    TranscriptionProcessor,
    VectorStorage,
    TextChunker,
    Config
)


async def basic_scraping_example():
    """Example of basic web scraping."""
    print("=== Basic Web Scraping Example ===")
    
    config = Config()
    
    async with ScrapingEngine(config.scraper) as engine:
        # Scrape a single URL
        result = await engine.scrape(
            "https://httpbin.org/html",
            options={
                'convert_to_markdown': True,
                'screenshot': False,
                'store_media': True
            }
        )
        
        print(f"Title: {result.title}")
        print(f"Success: {result.success}")
        print(f"Content length: {len(result.content)}")
        print(f"Markdown length: {len(result.markdown)}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Links found: {len(result.links)}")
        print(f"Media found: {len(result.media)}")


async def batch_scraping_example():
    """Example of batch scraping multiple URLs."""
    print("\n=== Batch Scraping Example ===")
    
    config = Config()
    
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/robots.txt"
    ]
    
    async with ScrapingEngine(config.scraper) as engine:
        results = await engine.scrape_batch(
            urls,
            options={
                'convert_to_markdown': True,
                'screenshot': False,
                'store_media': False
            }
        )
        
        successful = sum(1 for r in results if r.success)
        print(f"Batch scraping completed: {successful}/{len(results)} successful")
        
        for result in results:
            if result.success:
                print(f"✓ {result.url}: {len(result.content)} chars")
            else:
                print(f"✗ {result.url}: {result.error_message}")


async def website_crawling_example():
    """Example of website crawling."""
    print("\n=== Website Crawling Example ===")
    
    config = Config()
    
    async with WebCrawler(config.crawler) as crawler:
        # Start crawling a website
        job = await crawler.crawl_website(
            start_url="https://httpbin.org",
            max_depth=2,
            max_pages=10,
            stay_on_domain=True,
            respect_robots_txt=True
        )
        
        print(f"Started crawl job: {job.id}")
        print(f"Target domain: {job.domain}")
        
        # Monitor progress
        while job.status in ["pending", "running"]:
            await asyncio.sleep(2)
            job = crawler.get_job(job.id)
            print(f"Progress: {job.progress:.2%} - "
                  f"Discovered: {job.urls_discovered}, "
                  f"Crawled: {job.urls_crawled}, "
                  f"Successful: {job.urls_successful}")
        
        print(f"Crawl completed with status: {job.status}")
        print(f"Total results: {len(job.results)}")
        print(f"Total errors: {len(job.errors)}")


async def media_extraction_example():
    """Example of media extraction and processing."""
    print("\n=== Media Extraction Example ===")
    
    # Create a simple HTML with embedded media for testing
    test_html = """
    <html>
        <body>
            <h1>Test Page</h1>
            <img src="https://via.placeholder.com/150" alt="Test Image">
            <video controls>
                <source src="https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4" type="video/mp4">
            </video>
            <audio controls>
                <source src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" type="audio/mpeg">
            </audio>
        </body>
    </html>
    """
    
    # Extract media URLs from HTML
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(test_html, 'html.parser')
    
    media_urls = []
    for img in soup.find_all('img'):
        media_urls.append(img.get('src'))
    for video in soup.find_all(['video', 'source']):
        src = video.get('src')
        if src:
            media_urls.append(src)
    for audio in soup.find_all(['audio', 'source']):
        src = audio.get('src')
        if src:
            media_urls.append(src)
    
    print(f"Found {len(media_urls)} media URLs")
    
    # Process media
    async with MediaExtractor() as media_extractor:
        results = await media_extractor.extract_media_from_page(
            media_urls,
            process_images=True,
            process_videos=True,
            process_audio=True
        )
        
        print(f"Processed images: {len(results['images'])}")
        print(f"Processed videos: {len(results['videos'])}")
        print(f"Processed audio: {len(results['audio'])}")
        
        # Show OCR results for images
        for img_result in results['images']:
            if img_result.get('ocr_text'):
                print(f"Image OCR: {img_result['ocr_text'][:100]}...")
            
        # Show transcription results for audio
        for audio_result in results['audio']:
            if audio_result.get('transcription'):
                print(f"Audio transcription: {audio_result['transcription'][:100]}...")


async def vector_storage_example():
    """Example of vector storage and search."""
    print("\n=== Vector Storage Example ===")
    
    config = Config()
    
    # Sample documents
    documents = [
        {
            'id': 'doc1',
            'text': 'Python is a programming language that is widely used for web development, data analysis, and artificial intelligence.',
            'metadata': {'source': 'python_info', 'category': 'programming'}
        },
        {
            'id': 'doc2', 
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.',
            'metadata': {'source': 'ml_info', 'category': 'ai'}
        },
        {
            'id': 'doc3',
            'text': 'Web scraping is the process of extracting information from websites using automated tools.',
            'metadata': {'source': 'scraping_info', 'category': 'web_development'}
        }
    ]
    
    async with VectorStorage(config.vector_storage) as vector_storage:
        # Add documents
        doc_ids = await vector_storage.add_documents(documents)
        print(f"Added {len(doc_ids)} documents to vector storage")
        
        # Search for similar documents
        results = await vector_storage.search(
            query="programming and development",
            limit=5
        )
        
        print(f"Found {len(results)} similar documents:")
        for result in results:
            print(f"- {result['metadata']['category']}: {result['text'][:80]}... (score: {result['score']:.3f})")
        
        # Get collection stats
        stats = await vector_storage.get_collection_stats()
        print(f"Collection stats: {stats}")


async def text_chunking_example():
    """Example of text chunking for RAG applications."""
    print("\n=== Text Chunking Example ===")
    
    # Sample long text
    long_text = """
    Introduction to Machine Learning
    
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence (AI) and computer science, which focuses on 
    the use of data and algorithms to imitate the way that humans learn, gradually 
    improving its accuracy.
    
    Types of Machine Learning
    
    There are three main types of machine learning:
    
    1. Supervised Learning
    Supervised learning is when the algorithm learns from labeled training data. 
    The algorithm makes predictions or decisions by mapping input to output. 
    Examples include classification and regression tasks.
    
    2. Unsupervised Learning  
    Unsupervised learning finds hidden patterns in data without labeled examples. 
    The algorithm explores the data to discover patterns, groupings, or relationships.
    Examples include clustering and dimensionality reduction.
    
    3. Reinforcement Learning
    Reinforcement learning is when an agent learns to make decisions by taking actions 
    in an environment and receiving rewards or penalties. The goal is to learn the 
    optimal strategy for achieving the highest cumulative reward.
    
    Applications of Machine Learning
    
    Machine learning has many real-world applications:
    
    - Image Recognition: Identifying objects, faces, or scenes in images
    - Natural Language Processing: Understanding and generating human language
    - Recommendation Systems: Suggesting products, content, or services
    - Fraud Detection: Identifying suspicious transactions or activities
    - Medical Diagnosis: Assisting doctors in diagnosing diseases
    
    Future of Machine Learning
    
    Machine learning continues to evolve rapidly. With the advent of deep learning 
    and neural networks, the field has achieved breakthroughs in areas like computer 
    vision and natural language processing. The future promises even more sophisticated 
    AI systems that can understand, reason, and interact with humans more naturally.
    """
    
    chunker = TextChunker(
        chunk_size=500,
        chunk_overlap=100,
        chunking_strategy="semantic"
    )
    
    chunks = chunker.chunk_text(
        text=long_text,
        document_id="ml_intro",
        metadata={'source': 'tutorial', 'author': 'expert'}
    )
    
    print(f"Created {len(chunks)} chunks from text")
    
    # Show chunk statistics
    stats = chunker.get_chunking_stats(chunks)
    print(f"Average chunk size: {stats['average_chunk_size']:.0f} characters")
    print(f"Chunk count: {stats['total_chunks']}")
    
    # Show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk.text[:100]}...")
        print(f"Metadata: {chunk.metadata}")


async def complete_pipeline_example():
    """Example of a complete scraping and processing pipeline."""
    print("\n=== Complete Pipeline Example ===")
    
    config = Config()
    
    # URL to scrape
    url = "https://httpbin.org/html"
    
    try:
        async with ScrapingEngine(config.scraper) as engine:
            # Step 1: Scrape the page
            print(f"Step 1: Scraping {url}")
            result = await engine.scrape(url, {
                'convert_to_markdown': True,
                'screenshot': False,
                'store_media': True
            })
            
            if not result.success:
                print(f"Scraping failed: {result.error_message}")
                return
                
            # Step 2: Process text for RAG
            print("Step 2: Chunking text for RAG")
            chunker = TextChunker(chunk_size=300, chunking_strategy="semantic")
            chunks = chunker.chunk_text(
                result.markdown or result.content,
                document_id="scraped_page",
                metadata={
                    'url': result.url,
                    'title': result.title,
                    'scraped_at': result.timestamp.isoformat()
                }
            )
            
            print(f"Created {len(chunks)} chunks")
            
            # Step 3: Store in vector database
            print("Step 3: Storing in vector database")
            async with VectorStorage(config.vector_storage) as vector_storage:
                chunk_docs = [
                    {
                        'text': chunk.text,
                        'metadata': chunk.metadata
                    }
                    for chunk in chunks
                ]
                
                doc_ids = await vector_storage.add_documents(chunk_docs)
                print(f"Stored {len(doc_ids)} document chunks")
                
                # Step 4: Demonstrate search
                print("Step 4: Demonstrating semantic search")
                search_results = await vector_storage.search(
                    query="web page content",
                    limit=3
                )
                
                print(f"Found {len(search_results)} relevant chunks:")
                for i, result_chunk in enumerate(search_results):
                    print(f"{i+1}. {result_chunk['text'][:80]}... (score: {result_chunk['score']:.3f})")
                    
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")


async def main():
    """Run all examples."""
    print("Universal Web Scraper Examples")
    print("=" * 50)
    
    try:
        # Run individual examples
        await basic_scraping_example()
        await batch_scraping_example()
        await website_crawling_example()
        await text_chunking_example()
        await vector_storage_example()
        
        # Run complete pipeline
        await complete_pipeline_example()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())