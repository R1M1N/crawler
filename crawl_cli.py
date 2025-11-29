#!/usr/bin/env python3
"""
Depth-based Web Crawler CLI

Simple command-line interface for the depth-based web crawler.
Usage: python crawl_cli.py <url> --depth <number> [--output <file>]
"""

import asyncio
import argparse
from pathlib import Path
import json
from depth_crawler import DepthBasedCrawler


async def crawl_website(url: str, depth: int, output_file: str = None, concurrent_pages: int = 5):
    """
    Crawl a website with specified depth and save results.
    
    Args:
        url: Starting URL to crawl
        depth: Maximum depth (0=links on page only, 1=links+sublinks, etc.)
        output_file: Optional output file path
        concurrent_pages: Maximum concurrent pages to crawl
    """
    print(f"🕷️  Starting depth-based crawl of {url}")
    print(f"📊 Maximum depth: {depth}")
    print(f"🚀 Concurrent pages: {concurrent_pages}")
    print("=" * 60)
    
    # Initialize crawler
    crawler = DepthBasedCrawler(max_concurrent_pages=concurrent_pages)
    
    # Perform crawl
    try:
        results = await crawler.crawl_with_depth(url, max_depth=depth)
        
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            crawler.save_results(output_path, url, results)
            print(f"💾 Results saved to: {output_path}")
        
        # Display summary
        crawler.print_summary(results)
        
        # Show sample of discovered links
        if results['all_links']:
            print(f"\n🔗 SAMPLE OF DISCOVERED LINKS:")
            print("-" * 60)
            for i, link in enumerate(results['all_links'][:10]):  # Show first 10 links
                print(f"{i+1:2d}. {link.get('text', 'No text')[:50]:<50} → {link['url']}")
            
            if len(results['all_links']) > 10:
                print(f"    ... and {len(results['all_links']) - 10} more links")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during crawl: {e}")
        return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Depth-based Web Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl homepage only (depth=0)
  python crawl_cli.py https://example.com --depth 0
  
  # Crawl homepage + first level links (depth=1)
  python crawl_cli.py https://example.com --depth 1
  
  # Crawl with 2 levels and save results
  python crawl_cli.py https://example.com --depth 2 --output results.json
  
  # Crawl with more concurrent pages
  python crawl_cli.py https://example.com --depth 1 --concurrent 10
        """
    )
    
    parser.add_argument('url', help='Starting URL to crawl')
    parser.add_argument('--depth', '-d', type=int, default=0, 
                       help='Maximum depth (0=links on page only, 1=links+sublinks, etc.)')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output file path (JSON format)')
    parser.add_argument('--concurrent', '-c', type=int, default=5,
                       help='Maximum concurrent pages to crawl (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.depth < 0:
        print("❌ Error: Depth must be 0 or positive")
        return
    
    if args.concurrent < 1 or args.concurrent > 20:
        print("❌ Error: Concurrent pages must be between 1 and 20")
        return
    
    # Generate output filename if not specified
    if not args.output:
        url_domain = args.url.split('//')[1].split('/')[0] if '//' in args.url else args.url.split('/')[0]
        args.output = f"crawl_{url_domain}_depth{args.depth}.json"
    
    # Run crawl
    asyncio.run(crawl_website(
        url=args.url,
        depth=args.depth,
        output_file=args.output,
        concurrent_pages=args.concurrent
    ))


if __name__ == "__main__":
    main()