"""FastAPI server for the Universal Web Scraper."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from datetime import datetime
import logging

from .core.engine import ScrapingEngine
from .core.crawler import WebCrawler, CrawlJob
from .utils.config import Config, APIConfig
from .utils.logger import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="Universal Web Scraper API",
    description="Open-source alternative to Firecrawl for web scraping and crawling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
config = Config()
logger = get_logger("api")

# Global instances
scraping_engine: Optional[ScrapingEngine] = None
web_crawler: Optional[WebCrawler] = None
active_crawl_jobs: Dict[str, CrawlJob] = {}


# Pydantic models for API
class ScrapeRequest(BaseModel):
    url: HttpUrl
    options: Optional[Dict[str, Any]] = {
        'convert_to_markdown': True,
        'screenshot': False,
        'store_media': True
    }


class ScrapeResponse(BaseModel):
    id: str
    url: str
    title: str
    content: str
    markdown: str
    success: bool
    processing_time: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchScrapeRequest(BaseModel):
    urls: List[HttpUrl]
    options: Optional[Dict[str, Any]] = {
        'convert_to_markdown': True,
        'screenshot': False,
        'store_media': False
    }


class CrawlRequest(BaseModel):
    start_url: HttpUrl
    max_depth: Optional[int] = 3
    max_pages: Optional[int] = 100
    stay_on_domain: bool = True
    respect_robots_txt: bool = True
    url_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class CrawlResponse(BaseModel):
    job_id: str
    status: str
    start_url: str
    progress: float
    urls_discovered: int
    urls_crawled: int
    urls_successful: int
    urls_failed: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    urls_discovered: int
    urls_crawled: int
    urls_successful: int
    urls_failed: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results_count: int = 0
    errors_count: int = 0


# Dependency injection
async def get_scraping_engine():
    global scraping_engine
    if scraping_engine is None:
        scraping_engine = ScrapingEngine(config.scraper)
        await scraping_engine.start()
    return scraping_engine


async def get_web_crawler():
    global web_crawler
    if web_crawler is None:
        web_crawler = WebCrawler(config.crawler)
        await web_crawler.start()
    return web_crawler


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now().isoformat()
            }
        }
    )


# Routes
@app.get("/", summary="API Information")
async def root():
    """Get API information and status."""
    return {
        "name": "Universal Web Scraper API",
        "version": "1.0.0",
        "description": "Open-source web scraping and crawling API",
        "features": [
            "Single page scraping",
            "Website crawling", 
            "JavaScript rendering",
            "Media extraction",
            "OCR text extraction",
            "Audio/video transcription",
            "Vector storage for RAG",
            "Multiple output formats"
        ],
        "endpoints": {
            "scrape": "/scrape",
            "batch_scrape": "/scrape/batch", 
            "crawl": "/crawl",
            "job_status": "/crawl/{job_id}",
            "health": "/health",
            "stats": "/stats"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", summary="Health Check")
async def health_check():
    """Check API health status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "scraping_engine": "ready" if scraping_engine else "not_initialized",
            "web_crawler": "ready" if web_crawler else "not_initialized",
            "database": "connected"
        }
    }
    
    # Add warning if services not initialized
    if not scraping_engine or not web_crawler:
        health_status["status"] = "degraded"
        health_status["warnings"] = [
            "Services may not be fully initialized"
        ]
    
    return health_status


@app.post("/scrape", response_model=ScrapeResponse, summary="Scrape Single URL")
async def scrape_url(
    request: ScrapeRequest,
    engine: ScrapingEngine = Depends(get_scraping_engine)
):
    """
    Scrape a single URL and return extracted content.
    
    This endpoint scrapes a webpage and returns the content in multiple formats
    including Markdown, JSON, and HTML. It can handle JavaScript-rendered content
    and extract media information.
    """
    try:
        logger.info(f"Scraping URL: {request.url}")
        
        # Perform scraping
        result = await engine.scrape(str(request.url), request.options)
        
        # Format response
        response = ScrapeResponse(
            id=f"scrape_{int(datetime.now().timestamp())}",
            url=result.url,
            title=result.title,
            content=result.content,
            markdown=result.markdown,
            success=result.success,
            processing_time=result.processing_time,
            timestamp=result.timestamp.isoformat(),
            metadata=result.metadata,
            error=result.error_message
        )
        
        logger.info(f"Successfully scraped {result.url} in {result.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Failed to scrape {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/scrape/batch", summary="Batch Scrape URLs")
async def batch_scrape_urls(
    request: BatchScrapeRequest,
    background_tasks: BackgroundTasks,
    engine: ScrapingEngine = Depends(get_scraping_engine)
):
    """
    Scrape multiple URLs concurrently.
    
    This endpoint allows scraping multiple URLs at once for better efficiency.
    URLs are processed concurrently with rate limiting to be respectful.
    """
    try:
        urls = [str(url) for url in request.urls]
        logger.info(f"Batch scraping {len(urls)} URLs")
        
        # Perform batch scraping
        results = await engine.scrape_batch(urls, request.options)
        
        # Format responses
        responses = []
        for i, result in enumerate(results):
            response = ScrapeResponse(
                id=f"batch_scrape_{i}_{int(datetime.now().timestamp())}",
                url=result.url,
                title=result.title,
                content=result.content,
                markdown=result.markdown,
                success=result.success,
                processing_time=result.processing_time,
                timestamp=result.timestamp.isoformat(),
                metadata=result.metadata,
                error=result.error_message
            )
            responses.append(response)
            
        successful = sum(1 for r in responses if r.success)
        logger.info(f"Batch scraping completed: {successful}/{len(results)} successful")
        
        return {
            "results": responses,
            "summary": {
                "total_urls": len(urls),
                "successful": successful,
                "failed": len(results) - successful,
                "total_processing_time": sum(r.processing_time for r in responses)
            }
        }
        
    except Exception as e:
        logger.error(f"Batch scraping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scraping failed: {str(e)}")


@app.post("/crawl", response_model=CrawlResponse, summary="Start Website Crawl")
async def crawl_website(
    request: CrawlRequest,
    crawler: WebCrawler = Depends(get_web_crawler)
):
    """
    Start crawling a website.
    
    This endpoint initiates a website crawl that will discover and scrape
    multiple pages. The crawl runs in the background and you can check
    its status using the returned job_id.
    """
    try:
        logger.info(f"Starting crawl of {request.start_url}")
        
        # Start crawling
        job = await crawler.crawl_website(
            start_url=str(request.start_url),
            max_depth=request.max_depth,
            max_pages=request.max_pages,
            url_patterns=request.url_patterns,
            exclude_patterns=request.exclude_patterns,
            respect_robots_txt=request.respect_robots_txt
        )
        
        # Store job reference
        active_crawl_jobs[job.id] = job
        
        # Format response
        response = CrawlResponse(
            job_id=job.id,
            status=job.status,
            start_url=job.start_url,
            progress=job.progress,
            urls_discovered=job.urls_discovered,
            urls_crawled=job.urls_crawled,
            urls_successful=job.urls_successful,
            urls_failed=job.urls_failed,
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None
        )
        
        logger.info(f"Started crawl job {job.id} for {job.start_url}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to start crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl initiation failed: {str(e)}")


@app.get("/crawl/{job_id}/status", response_model=JobStatusResponse, summary="Get Crawl Job Status")
async def get_crawl_job_status(
    job_id: str,
    crawler: WebCrawler = Depends(get_web_crawler)
):
    """Get the status of a crawl job."""
    try:
        job = crawler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
            
        response = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            urls_discovered=job.urls_discovered,
            urls_crawled=job.urls_crawled,
            urls_successful=job.urls_successful,
            urls_failed=job.urls_failed,
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            results_count=len(job.results),
            errors_count=len(job.errors)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crawl job status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/crawl/{job_id}/results", summary="Get Crawl Job Results")
async def get_crawl_job_results(
    job_id: str,
    crawler: WebCrawler = Depends(get_web_crawler),
    limit: int = 100,
    offset: int = 0
):
    """Get results from a completed crawl job."""
    try:
        job = crawler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
            
        if job.status != "completed":
            raise HTTPException(status_code=400, detail=f"Crawl job not completed (status: {job.status})")
            
        # Get paginated results
        results_slice = job.results[offset:offset + limit]
        
        # Format results
        formatted_results = []
        for result in results_slice:
            formatted_results.append({
                "url": result.url,
                "title": result.title,
                "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                "success": result.success,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            })
            
        return {
            "job_id": job_id,
            "results": formatted_results,
            "pagination": {
                "total": len(job.results),
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < len(job.results)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get crawl results: {e}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")


@app.delete("/crawl/{job_id}", summary="Stop/Delete Crawl Job")
async def delete_crawl_job(
    job_id: str,
    crawler: WebCrawler = Depends(get_web_crawler)
):
    """Stop and delete a crawl job."""
    try:
        job = crawler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
            
        # Stop job if running
        if job.status == "running":
            await crawler.stop_job(job_id)
            
        # Delete job
        success = await crawler.delete_job(job_id)
        if success:
            active_crawl_jobs.pop(job_id, None)
            
        return {
            "message": f"Crawl job {job_id} deleted successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete crawl job: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.get("/jobs", summary="List All Crawl Jobs")
async def list_crawl_jobs(
    crawler: WebCrawler = Depends(get_web_crawler),
    status: Optional[str] = None,
    limit: int = 50
):
    """List all crawl jobs."""
    try:
        jobs = crawler.list_jobs()
        
        # Filter by status if requested
        if status:
            jobs = [job for job in jobs if job.status == status]
            
        # Format responses
        job_responses = []
        for job in jobs[:limit]:
            response = JobStatusResponse(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                urls_discovered=job.urls_discovered,
                urls_crawled=job.urls_crawled,
                urls_successful=job.urls_successful,
                urls_failed=job.urls_failed,
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                results_count=len(job.results),
                errors_count=len(job.errors)
            )
            job_responses.append(response)
            
        return {
            "jobs": job_responses,
            "total": len(jobs),
            "showing": len(job_responses)
        }
        
    except Exception as e:
        logger.error(f"Failed to list crawl jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@app.get("/stats", summary="Get API Statistics")
async def get_api_stats(
    crawler: WebCrawler = Depends(get_web_crawler)
):
    """Get API usage statistics."""
    try:
        # Get crawl job stats
        jobs = crawler.list_jobs()
        job_stats = {
            "total_jobs": len(jobs),
            "running_jobs": len([j for j in jobs if j.status == "running"]),
            "completed_jobs": len([j for j in jobs if j.status == "completed"]),
            "failed_jobs": len([j for j in jobs if j.status == "failed"])
        }
        
        # Calculate totals
        total_urls_discovered = sum(job.urls_discovered for job in jobs)
        total_urls_crawled = sum(job.urls_crawled for job in jobs)
        total_urls_successful = sum(job.urls_successful for job in jobs)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "crawl_jobs": job_stats,
            "url_stats": {
                "total_discovered": total_urls_discovered,
                "total_crawled": total_urls_crawled,
                "total_successful": total_urls_successful,
                "success_rate": (total_urls_successful / max(total_urls_crawled, 1)) * 100
            },
            "active_jobs": len(active_crawl_jobs)
        }
        
    except Exception as e:
        logger.error(f"Failed to get API stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Universal Web Scraper API...")
    
    # Initialize global instances
    global scraping_engine, web_crawler
    
    try:
        scraping_engine = ScrapingEngine(config.scraper)
        await scraping_engine.start()
        logger.info("Scraping engine initialized")
        
        web_crawler = WebCrawler(config.crawler)
        await web_crawler.start()
        logger.info("Web crawler initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Universal Web Scraper API...")
    
    global scraping_engine, web_crawler
    
    try:
        if web_crawler:
            await web_crawler.stop()
        if scraping_engine:
            await scraping_engine.stop()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
        reload=config.api.reload,
        log_level="info"
    )