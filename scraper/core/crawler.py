"""
Universal Web Crawler

Advanced web crawler that can systematically explore entire websites
with configurable depth, URL filtering, and polite crawling policies.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser

import aiohttp
from bs4 import BeautifulSoup
from tld import get_tld

from .engine import ScrapingEngine, ScrapingResult
from ..utils.config import CrawlerConfig
from ..utils.logger import get_logger
from ..utils.validators import URLValidator, URLFilter
from ..utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


class CrawlJob:
    """Represents a crawling job with its configuration and progress."""
    
    def __init__(self, start_url: str, config: CrawlerConfig):
        self.id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_url = start_url
        self.config = config
        self.status = "pending"  # pending, running, completed, failed, stopped
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0.0
        self.urls_discovered = 0
        self.urls_crawled = 0
        self.urls_successful = 0
        self.urls_failed = 0
        self.results: List[ScrapingResult] = []
        self.errors: List[str] = []
        self.domain = self._extract_domain(start_url)
        
    def _extract_domain(self, url: str) -> str:
        """Extract the main domain from URL."""
        try:
            return get_tld(url, as_object=True).fld
        except Exception:
            return urlparse(url).netloc
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert crawl job to dictionary for API responses."""
        return {
            'id': self.id,
            'start_url': self.start_url,
            'domain': self.domain,
            'status': self.status,
            'progress': self.progress,
            'urls_discovered': self.urls_discovered,
            'urls_crawled': self.urls_crawled,
            'urls_successful': self.urls_successful,
            'urls_failed': self.urls_failed,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_results': len(self.results),
            'total_errors': len(self.errors)
        }


class WebCrawler:
    """Advanced web crawler with comprehensive crawling capabilities."""
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.engine: Optional[ScrapingEngine] = None
        self.url_validator = URLValidator()
        self.url_filter = URLFilter()
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.requests_per_second,
            delay_between_requests=self.config.delay_between_requests
        )
        
        # Active crawl jobs
        self.active_jobs: Dict[str, CrawlJob] = {}
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.engine = ScrapingEngine(self.config.scraping_config)
        await self.engine.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.engine:
            await self.engine.stop()
            
    async def crawl_website(
        self, 
        start_url: str, 
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        url_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        respect_robots_txt: bool = True,
        custom_headers: Optional[Dict[str, str]] = None
    ) -> CrawlJob:
        """
        Start crawling a website from the given URL.
        
        Args:
            start_url: URL to start crawling from
            max_depth: Maximum crawling depth (None for unlimited)
            max_pages: Maximum number of pages to crawl (None for unlimited)
            url_patterns: List of URL patterns to include (regex)
            exclude_patterns: List of URL patterns to exclude (regex)
            respect_robots_txt: Whether to respect robots.txt
            custom_headers: Additional headers to use for requests
            
        Returns:
            CrawlJob object representing the crawling job
        """
        if not self.url_validator.is_valid(start_url):
            raise ValueError(f"Invalid start URL: {start_url}")
            
        job = CrawlJob(start_url, self.config)
        self.active_jobs[job.id] = job
        
        # Update config with provided parameters
        if max_depth is not None:
            job.config.max_depth = max_depth
        if max_pages is not None:
            job.config.max_pages = max_pages
        if url_patterns is not None:
            job.config.url_patterns = url_patterns
        if exclude_patterns is not None:
            job.config.exclude_patterns = exclude_patterns
        if custom_headers is not None:
            job.config.scraping_config.custom_headers = custom_headers
            
        # Start crawling in background
        asyncio.create_task(self._crawl_job_worker(job, respect_robots_txt))
        
        return job
        
    async def _crawl_job_worker(self, job: CrawlJob, respect_robots_txt: bool):
        """Worker function that performs the actual crawling."""
        job.status = "running"
        job.started_at = datetime.now()
        
        try:
            logger.info(f"Starting crawl job {job.id} for {job.start_url}")
            
            # Check robots.txt if required
            robots_allowed = True
            if respect_robots_txt:
                robots_allowed = await self._check_robots_txt(job.start_url)
                if not robots_allowed:
                    job.status = "failed"
                    job.errors.append("Crawling disallowed by robots.txt")
                    return
                    
            # Discover initial URLs
            discovered_urls = await self._discover_urls([job.start_url], job)
            
            # Add start URL if not discovered
            if job.start_url not in discovered_urls:
                discovered_urls.add(job.start_url)
                
            job.urls_discovered = len(discovered_urls)
            logger.info(f"Discovered {job.urls_discovered} URLs for crawling")
            
            # Start crawling with BFS
            await self._crawl_urls_bfs(discovered_urls, job)
            
            job.status = "completed"
            job.completed_at = datetime.now()
            logger.info(f"Crawl job {job.id} completed successfully")
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now()
            job.errors.append(str(e))
            logger.error(f"Crawl job {job.id} failed: {e}")
            
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if crawling is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            user_agent = self.config.user_agent
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning(f"Failed to check robots.txt for {url}: {e}")
            return True  # Default to allowing if robots.txt check fails
            
    async def _discover_urls(self, urls: List[str], job: CrawlJob) -> Set[str]:
        """Discover URLs from the given URLs."""
        discovered = set()
        discovered_urls = set(urls)
        
        # Limit concurrent discovery requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def discover_from_url(url: str) -> Set[str]:
            async with semaphore:
                await self.rate_limiter.acquire()
                
                try:
                    # Simple HTTP request to get the page
                    async with aiohttp.ClientSession(
                        headers={'User-Agent': self.config.user_agent},
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as session:
                        async with session.get(url) as response:
                            if response.status != 200:
                                return set()
                                
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract links
                            urls = set()
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                absolute_url = urljoin(url, href)
                                
                                # Filter URLs
                                if self._should_crawl_url(absolute_url, job):
                                    urls.add(absolute_url)
                                    
                            return urls
                            
                except Exception as e:
                    logger.debug(f"Failed to discover URLs from {url}: {e}")
                    return set()
                    
        # Discover URLs from all input URLs
        tasks = [discover_from_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, set):
                discovered.update(result)
                
        return discovered
        
    async def _crawl_urls_bfs(self, urls: Set[str], job: CrawlJob):
        """Crawl URLs using breadth-first search."""
        crawled = set()
        to_crawl = list(urls)
        depth = 0
        
        while to_crawl and (job.config.max_depth is None or depth <= job.config.max_depth):
            if job.config.max_pages and len(crawled) >= job.config.max_pages:
                break
                
            current_depth_urls = []
            for _ in range(min(len(to_crawl), job.config.max_concurrent_requests)):
                if not to_crawl:
                    break
                current_depth_urls.append(to_crawl.pop(0))
                
            if not current_depth_urls:
                break
                
            # Crawl URLs at current depth
            await self._crawl_url_batch(current_depth_urls, job, crawled)
            
            # Discover next level URLs
            if depth < (job.config.max_depth or float('inf')):
                new_urls = await self._discover_urls(list(crawled - set(current_depth_urls)), job)
                # Add only new URLs not yet discovered
                to_crawl.extend(new_urls - crawled - set(current_depth_urls))
                
            depth += 1
            
    async def _crawl_url_batch(self, urls: List[str], job: CrawlJob, crawled: Set[str]):
        """Crawl a batch of URLs concurrently."""
        if not self.engine:
            raise RuntimeError("Scraping engine not initialized")
            
        # Limit concurrent crawling
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def crawl_single_url(url: str):
            async with semaphore:
                await self.rate_limiter.acquire()
                
                try:
                    # Scrape the URL
                    result = await self.engine.scrape(url, {
                        'convert_to_markdown': True,
                        'screenshot': False,  # Disable screenshots during crawling for performance
                        'store_media': False  # Don't store media during crawling
                    })
                    
                    job.results.append(result)
                    crawled.add(url)
                    job.urls_crawled += 1
                    
                    if result.success:
                        job.urls_successful += 1
                    else:
                        job.urls_failed += 1
                        
                    # Update progress
                    total_planned = job.urls_discovered
                    if total_planned > 0:
                        job.progress = min(len(crawled) / total_planned, 1.0)
                        
                    logger.debug(f"Crawled {url}: {'Success' if result.success else 'Failed'}")
                    
                except Exception as e:
                    job.urls_failed += 1
                    job.errors.append(f"Failed to crawl {url}: {e}")
                    logger.error(f"Error crawling {url}: {e}")
                    
        # Execute batch crawling
        await asyncio.gather(*[crawl_single_url(url) for url in urls])
        
    def _should_crawl_url(self, url: str, job: CrawlJob) -> bool:
        """Determine if a URL should be crawled based on filters and constraints."""
        # Basic validation
        if not self.url_validator.is_valid(url):
            return False
            
        # Check domain constraint (stay on same domain)
        if job.config.stay_on_domain:
            url_domain = self._extract_domain(url)
            if url_domain != job.domain:
                return False
                
        # Check URL patterns
        if job.config.url_patterns:
            if not self.url_filter.matches_any_pattern(url, job.config.url_patterns):
                return False
                
        # Check exclude patterns
        if job.config.exclude_patterns:
            if self.url_filter.matches_any_pattern(url, job.config.exclude_patterns):
                return False
                
        # Check file extensions
        if job.config.allowed_extensions:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            if not any(path.endswith(ext) for ext in job.config.allowed_extensions):
                return False
                
        # Check excluded extensions
        if job.config.excluded_extensions:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            if any(path.endswith(ext) for ext in job.config.excluded_extensions):
                return False
                
        return True
        
    def _extract_domain(self, url: str) -> str:
        """Extract the main domain from URL."""
        try:
            return get_tld(url, as_object=True).fld
        except Exception:
            return urlparse(url).netloc
            
    def get_job(self, job_id: str) -> Optional[CrawlJob]:
        """Get a crawl job by ID."""
        return self.active_jobs.get(job_id)
        
    def list_jobs(self) -> List[CrawlJob]:
        """List all crawl jobs."""
        return list(self.active_jobs.values())
        
    async def stop_job(self, job_id: str) -> bool:
        """Stop a running crawl job."""
        job = self.active_jobs.get(job_id)
        if job and job.status == "running":
            job.status = "stopped"
            job.completed_at = datetime.now()
            logger.info(f"Stopped crawl job {job_id}")
            return True
        return False
        
    async def delete_job(self, job_id: str) -> bool:
        """Delete a crawl job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status == "running":
                await self.stop_job(job_id)
            del self.active_jobs[job_id]
            logger.info(f"Deleted crawl job {job_id}")
            return True
        return False