"""
Universal Web Scraper Engine

Core scraping engine using Playwright for handling JavaScript-heavy sites
and modern web applications. Provides multiple output formats and extraction strategies.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from bs4 import BeautifulSoup
import aiohttp
from aiohttp_socks import ProxyConnector
from readability import Document
import markdown
from PIL import Image
import base64
import io

from ..utils.config import ScraperConfig
from ..utils.logger import get_logger
from ..utils.storage import StorageManager
from ..utils.validators import URLValidator
from ..types import ScrapingResult, ScrapedPage, MediaContent, ExtractedLink

logger = get_logger(__name__)



class ScrapingEngine:
    """Main scraping engine with advanced capabilities."""
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.storage = StorageManager()
        self.url_validator = URLValidator()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        
    async def start(self):
        """Initialize the browser and context."""
        self.playwright = await async_playwright().start()
        
        # Browser arguments for stealth
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-features=IsolateOrigins,site-per-process',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu',
        ]
        
        self.browser = await self.playwright.chromium.launch(
            headless=self.config.headless,
            args=browser_args,
            proxy=self.config.proxy if self.config.proxy else None
        )
        
        self.context = await self.browser.new_context(
            user_agent=self.config.user_agent,
            viewport={'width': 1920, 'height': 1080},
            java_script_enabled=True
        )
        
        logger.info("Scraping engine started successfully")
        
    async def stop(self):
        """Close the browser and context."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("Scraping engine stopped")
        
    async def scrape(self, url: str, options: Optional[Dict] = None) -> ScrapingResult:
        """
        Scrape a single URL and return comprehensive results.
        
        Args:
            url: URL to scrape
            options: Additional scraping options
            
        Returns:
            ScrapingResult object with all extracted data
        """
        start_time = datetime.now()
        result = ScrapingResult()
        result.url = url
        
        options = options or {}
        
        try:
            # Validate URL
            if not self.url_validator.is_valid(url):
                raise ValueError(f"Invalid URL: {url}")
                
            logger.info(f"Starting scrape of {url}")
            
            # Create new page for this scrape
            page = await self.context.new_page()
            
            # Set additional headers
            if self.config.custom_headers:
                await page.set_extra_http_headers(self.config.custom_headers)
                
            # Navigate to page
            response = await page.goto(
                url,
                wait_until='networkidle',
                timeout=self.config.timeout * 1000
            )
            
            if not response:
                raise Exception("Failed to load page")
                
            result.status_code = response.status
            result.headers = dict(response.headers)
            
            # Smart wait for dynamic content
            if options.get('wait_for_selector'):
                try:
                    await page.wait_for_selector(
                        options['wait_for_selector'],
                        timeout=10000
                    )
                except Exception:
                    logger.warning(f"Selector {options['wait_for_selector']} not found")
            
            # Execute additional actions if provided
            if options.get('actions'):
                await self._execute_actions(page, options['actions'])
                
            # Extract content
            await self._extract_content(page, result, options)
            
            # Take screenshot if requested
            if options.get('screenshot', False):
                result.screenshot = await page.screenshot(
                    full_page=options.get('full_page_screenshot', True),
                    type='png'
                )
                
            # Store results
            await self._store_results(result, options)
            
            result.success = True
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully scraped {url} in {result.processing_time:.2f}s")
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
            result.processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to scrape {url}: {e}")
            
        finally:
            if 'page' in locals():
                await page.close()
                
        return result
        
    async def _extract_content(self, page: Page, result: ScrapingResult, options: Dict):
        """Extract all content from the page."""
        
        # Extract basic page info
        result.title = await page.title()
        
        # Extract HTML content
        html_content = await page.content()
        result.html = html_content
        
        # Parse with BeautifulSoup for structured extraction
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract main content using readability
        try:
            readable_doc = Document(html_content)
            readable_html = readable_doc.summary()
            result.content = BeautifulSoup(readable_html, 'html.parser').get_text()
        except Exception:
            # Fallback to basic text extraction
            for script in soup(["script", "style"]):
                script.decompose()
            result.content = soup.get_text()
            
        # Convert to markdown
        if options.get('convert_to_markdown', True):
            result.markdown = self._html_to_markdown(result.html)
            
        # Extract metadata
        result.metadata = self._extract_metadata(soup)
        
        # Extract links
        result.links = self._extract_links(soup, result.url)
        
        # Extract media
        result.media = await self._extract_media(page, soup, result.url)
        
    async def _execute_actions(self, page: Page, actions: List[Dict]):
        """Execute browser actions like clicking, scrolling, typing."""
        for action in actions:
            action_type = action.get('type')
            
            if action_type == 'wait':
                await page.wait_for_timeout(action.get('duration', 1000))
                
            elif action_type == 'click':
                selector = action.get('selector')
                if selector:
                    await page.click(selector)
                    
            elif action_type == 'scroll':
                await page.evaluate(f"window.scrollTo(0, {action.get('y', 0)})")
                
            elif action_type == 'type':
                selector = action.get('selector')
                text = action.get('text')
                if selector and text:
                    await page.fill(selector, text)
                    
            elif action_type == 'press':
                selector = action.get('selector')
                key = action.get('key')
                if selector and key:
                    await page.press(selector, key)
                    
        # Wait for any dynamic content to load
        await page.wait_for_timeout(2000)
        
    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown using readability."""
        try:
            doc = Document(html)
            markdown_text = markdown.markdown(doc.summary())
            return markdown_text
        except Exception:
            # Fallback to basic HTML to text conversion
            return BeautifulSoup(html, 'html.parser').get_text()
            
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = {}
        
        # Standard meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
                
        # Open Graph tags
        for meta in soup.find_all('meta', property=True):
            property_name = meta.get('property')
            content = meta.get('content')
            if property_name and content:
                metadata[f"og:{property_name}"] = content
                
        # Schema.org JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            try:
                import json
                metadata['schema_org'] = json.loads(json_ld_scripts[0].string)
            except Exception:
                pass
                
        return metadata
        
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            if self.url_validator.is_valid(absolute_url):
                links.append(absolute_url)
        return list(set(links))  # Remove duplicates
        
    async def _extract_media(self, page: Page, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract media information from the page."""
        media = []
        
        # Images
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                absolute_url = urljoin(base_url, src)
                media.append({
                    'type': 'image',
                    'url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width'),
                    'height': img.get('height')
                })
                
        # Videos
        for video in soup.find_all(['video', 'source']):
            src = video.get('src')
            if src:
                absolute_url = urljoin(base_url, src)
                media.append({
                    'type': 'video',
                    'url': absolute_url,
                    'poster': video.get('poster', ''),
                    'duration': video.get('duration')
                })
                
        # Audio
        for audio in soup.find_all(['audio', 'source']):
            src = audio.get('src')
            if src:
                absolute_url = urljoin(base_url, src)
                media.append({
                    'type': 'audio',
                    'url': absolute_url,
                    'duration': audio.get('duration')
                })
                
        return media
        
    async def _store_results(self, result: ScrapingResult, options: Dict):
        """Store scraping results based on configuration."""
        if not self.config.storage_enabled:
            return
            
        storage_options = {
            'format': options.get('storage_format', 'json'),
            'include_screenshot': options.get('include_screenshot', False),
            'store_media': options.get('store_media', True)
        }
        
        await self.storage.store_scraping_result(result, storage_options)
        
    async def scrape_batch(self, urls: List[str], options: Optional[Dict] = None) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            options: Additional scraping options
            
        Returns:
            List of ScrapingResult objects
        """
        logger.info(f"Starting batch scrape of {len(urls)} URLs")
        
        # Limit concurrent requests based on config
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape(url, options)
                
        results = await asyncio.gather(
            *[scrape_with_semaphore(url) for url in urls],
            return_exceptions=True
        )
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape {urls[i]}: {result}")
            else:
                valid_results.append(result)
                
        logger.info(f"Batch scrape completed: {len(valid_results)}/{len(urls)} successful")
        return valid_results