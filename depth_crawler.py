#!/usr/bin/env python3
"""
Depth-based web crawler implementation.

This implements the depth-based crawling system where:
- depth=0: all links in the same page
- depth=1: every link on that page + links within those links
- depth=2: sub to sub page
- etc.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse
import json
import logging

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from bs4 import BeautifulSoup
from readability import Document
import markdown


logger = logging.getLogger(__name__)


class DepthBasedCrawler:
    """
    Depth-based web crawler that discovers and scrapes content level by level.
    
    Depth definitions:
    - depth=0: extract all links from the starting page only
    - depth=1: extract links from starting page + all pages found at depth 0
    - depth=2: extract links from all pages found at depth 0 and 1
    - etc.
    """
    
    def __init__(self, max_concurrent_pages: int = 5):
        self.max_concurrent_pages = max_concurrent_pages
        self.visited_urls: Set[str] = set()
        self.discovered_links: List[Dict[str, Any]] = []
        self.crawl_results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    async def crawl_with_depth(self, start_url: str, max_depth: int = 0) -> Dict[str, Any]:
        """
        Crawl a website using depth-based discovery.
        
        Args:
            start_url: The starting URL to crawl
            max_depth: Maximum depth to crawl (0 = links on same page only)
            
        Returns:
            Dictionary containing crawl results and statistics
        """
        self.start_time = datetime.now()
        self.visited_urls.clear()
        self.discovered_links.clear()
        self.crawl_results.clear()
        
        logger.info(f"Starting depth-based crawl of {start_url} with max_depth={max_depth}")
        
        # Level-based BFS traversal
        current_level_urls = {start_url}
        
        for depth in range(max_depth + 1):
            logger.info(f"Crawling depth {depth}: {len(current_level_urls)} pages")
            
            # Crawl all URLs at current depth
            if current_level_urls:
                await self._crawl_level(current_level_urls, depth)
            
            # Prepare URLs for next level (if not at max depth)
            if depth < max_depth:
                current_level_urls = self._get_next_level_urls()
                if not current_level_urls:
                    logger.info(f"No more pages to crawl. Stopping at depth {depth}")
                    break
        
        self.end_time = datetime.now()
        
        return self._compile_results()
    
    async def _crawl_level(self, urls: Set[str], depth: int):
        """Crawl a set of URLs at the same depth level."""
        # Limit concurrent pages
        semaphore = asyncio.Semaphore(self.max_concurrent_pages)
        
        async def crawl_single_url(url: str):
            async with semaphore:
                await self._crawl_single_page(url, depth)
        
        # Execute concurrent crawling
        await asyncio.gather(*[crawl_single_url(url) for url in urls])
    
    async def _crawl_single_page(self, url: str, depth: int):
        """Crawl a single page and extract all links and content."""
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent and other settings
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                # Navigate to page
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Get page content
                html_content = await page.content()
                title = await page.title()
                
                # Parse and extract links
                soup = BeautifulSoup(html_content, 'html.parser')
                page_links = self._extract_links(soup, url)
                
                # Store page information
                page_info = {
                    'url': url,
                    'title': title,
                    'depth': depth,
                    'links_found': len(page_links),
                    'timestamp': datetime.now().isoformat(),
                    'links': page_links
                }
                
                self.crawl_results.append(page_info)
                self.discovered_links.extend(page_links)
                
                logger.info(f"Depth {depth}: {url} - {len(page_links)} links found")
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            page_info = {
                'url': url,
                'title': None,
                'depth': depth,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'links': []
            }
            self.crawl_results.append(page_info)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract all links from a page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True)
            title = link.get('title', '')
            
            # Convert relative URLs to absolute
            if not (href.startswith('http://') or href.startswith('https://')):
                href = urljoin(base_url, href)
            
            # Filter out unwanted links
            if self._is_valid_link(href):
                link_info = {
                    'url': href,
                    'text': text,
                    'title': title,
                    'source_page': base_url
                }
                links.append(link_info)
        
        return links
    
    def _is_valid_link(self, url: str) -> bool:
        """Check if a link is valid for crawling."""
        # Skip certain file types and protocols
        invalid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml', '.zip'}
        invalid_protocols = {'mailto:', 'tel:', 'ftp:', 'file:'}
        
        # Check protocol
        for protocol in invalid_protocols:
            if url.startswith(protocol):
                return False
        
        # Check extension
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in invalid_extensions:
            if path.endswith(ext):
                return False
        
        return True
    
    def _get_next_level_urls(self) -> Set[str]:
        """Get unique URLs for the next depth level."""
        next_level_urls = set()
        
        for link_info in self.discovered_links:
            link_url = link_info['url']
            if link_url not in self.visited_urls and self._is_valid_link(link_url):
                next_level_urls.add(link_url)
        
        return next_level_urls
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final crawl results."""
        if not self.start_time or not self.end_time:
            duration = None
        else:
            duration = (self.end_time - self.start_time).total_seconds()
        
        # Group results by depth
        results_by_depth = {}
        for result in self.crawl_results:
            depth = result.get('depth', 0)
            if depth not in results_by_depth:
                results_by_depth[depth] = []
            results_by_depth[depth].append(result)
        
        # Calculate statistics
        total_pages = len(self.crawl_results)
        total_links = len(self.discovered_links)
        successful_pages = len([r for r in self.crawl_results if 'error' not in r])
        failed_pages = len([r for r in self.crawl_results if 'error' in r])
        
        return {
            'summary': {
                'start_url': None,  # Will be set by caller
                'max_depth_reached': max([r.get('depth', 0) for r in self.crawl_results]) if self.crawl_results else 0,
                'total_pages_crawled': total_pages,
                'successful_pages': successful_pages,
                'failed_pages': failed_pages,
                'total_links_discovered': total_links,
                'duration_seconds': duration,
                'crawl_started': self.start_time.isoformat() if self.start_time else None,
                'crawl_completed': self.end_time.isoformat() if self.end_time else None,
            },
            'results_by_depth': results_by_depth,
            'all_results': self.crawl_results,
            'all_links': self.discovered_links,
        }
    
    def save_results(self, filepath: Path, start_url: str, results: Dict[str, Any]):
        """Save crawl results to JSON file."""
        results['summary']['start_url'] = start_url
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Crawl results saved to {filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of crawl results."""
        summary = results['summary']
        results_by_depth = results['results_by_depth']
        
        print("\n" + "=" * 80)
        print("🔍 DEPTH-BASED CRAWL RESULTS")
        print("=" * 80)
        print(f"🌐 Start URL: {summary['start_url']}")
        print(f"📊 Max Depth Reached: {summary['max_depth_reached']}")
        print(f"📄 Pages Crawled: {summary['total_pages_crawled']}")
        print(f"✅ Successful: {summary['successful_pages']}")
        print(f"❌ Failed: {summary['failed_pages']}")
        print(f"🔗 Total Links Discovered: {summary['total_links_discovered']}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.2f} seconds")
        print("=" * 80)
        
        print("\n📈 RESULTS BY DEPTH:")
        print("-" * 50)
        
        for depth in sorted(results_by_depth.keys()):
            depth_results = results_by_depth[depth]
            depth_pages = len(depth_results)
            depth_links = sum(r.get('links_found', 0) for r in depth_results)
            depth_errors = len([r for r in depth_results if 'error' in r])
            
            print(f"Depth {depth}: {depth_pages} pages, {depth_links} links, {depth_errors} errors")
        
        # Show sample links from depth 0 (top-level links)
        if 0 in results_by_depth:
            depth_0_results = results_by_depth[0]
            print(f"\n🔗 SAMPLE LINKS FROM DEPTH 0:")
            print("-" * 50)
            for i, result in enumerate(depth_0_results[:3]):  # Show first 3 pages
                print(f"\nPage {i+1}: {result.get('title', 'No title')}")
                print(f"URL: {result['url']}")
                print(f"Links found: {result.get('links_found', 0)}")
                
                # Show first few links
                for link in result.get('links', [])[:3]:
                    print(f"  - {link.get('text', 'No text')}: {link['url']}")
        
        print("\n" + "=" * 80)


async def test_depth_based_crawling():
    """Test the depth-based crawling system."""
    print("🧪 Testing Depth-Based Crawling System")
    print("=" * 60)
    
    crawler = DepthBasedCrawler(max_concurrent_pages=3)
    
    # Test with Hugging Face at depth 0 (links on same page only)
    print("\n🚀 Testing Hugging Face with depth=0...")
    results = await crawler.crawl_with_depth("https://huggingface.co/", max_depth=0)
    
    # Save and display results
    results_file = Path("huggingface_depth0_results.json")
    crawler.save_results(results_file, "https://huggingface.co/", results)
    crawler.print_summary(results)
    
    # Test with depth=1 (one level deeper)
    print("\n🚀 Testing Hugging Face with depth=1...")
    crawler2 = DepthBasedCrawler(max_concurrent_pages=2)  # Limit to avoid overloading
    
    results2 = await crawler2.crawl_with_depth("https://huggingface.co/", max_depth=1)
    
    results2_file = Path("huggingface_depth1_results.json")
    crawler2.save_results(results2_file, "https://huggingface.co/", results2)
    crawler2.print_summary(results2)
    
    print("\n🎉 Depth-based crawling tests completed!")


if __name__ == "__main__":
    asyncio.run(test_depth_based_crawling())