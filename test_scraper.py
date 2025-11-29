#!/usr/bin/env python3
"""
Simple test to verify the basic scraping functionality works.
This test bypasses the complex import structure.
"""

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from readability import Document
import markdown


async def test_basic_scraping():
    """Test basic scraping functionality."""
    print("🕷️  Testing Universal Web Scraper...")
    
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            print("✅ Browser launched successfully")
            
            # Navigate to example.com
            print("📡 Navigating to https://example.com...")
            await page.goto('https://example.com')
            
            # Wait for page to load
            await page.wait_for_load_state('networkidle')
            
            # Get HTML content
            html_content = await page.content()
            
            print("✅ Page loaded successfully")
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.title.text if soup.title else "No title found"
            
            # Extract main content using readability
            doc = Document(html_content)
            main_content = doc.summary()
            
            # Convert to markdown
            md = markdown.Markdown()
            markdown_content = md.convert(main_content)
            
            # Extract basic info
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            images = [img.get('src') for img in soup.find_all('img', src=True)]
            
            await browser.close()
            
            # Print results
            print("\n📊 SCRAPING RESULTS")
            print("=" * 50)
            print(f"🌐 URL: https://example.com")
            print(f"📑 Title: {title}")
            print(f"🔗 Links found: {len(links)}")
            print(f"🖼️  Images found: {len(images)}")
            print(f"📝 Content preview: {markdown_content[:200]}...")
            print("=" * 50)
            
            print("\n🎉 Basic scraping test PASSED!")
            return True
            
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        return False


async def test_huggingface_scraping():
    """Test scraping Hugging Face website."""
    print("\n🚀 Testing Hugging Face scraping...")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            print("📡 Navigating to https://huggingface.co/")
            await page.goto('https://huggingface.co/', wait_until='networkidle', timeout=30000)
            
            # Get HTML content
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract basic info
            title = soup.title.text if soup.title else "No title found"
            links = [a.get('href') for a in soup.find_all('a', href=True)][:10]  # First 10 links
            nav_links = [a.get('href') for a in soup.find_all('a', href=True) if 'href' in str(a.parent) and 'nav' in str(a.parent).lower()]
            
            await browser.close()
            
            # Print results
            print("\n📊 HUGGING FACE SCRAPING RESULTS")
            print("=" * 50)
            print(f"🌐 URL: https://huggingface.co/")
            print(f"📑 Title: {title}")
            print(f"🔗 Sample links found: {len(links)}")
            print("🔗 Navigation links found:")
            for link in nav_links[:5]:
                print(f"  - {link}")
            print(f"📝 Content preview: {soup.get_text()[:200]}...")
            print("=" * 50)
            
            print("\n✅ Hugging Face scraping test PASSED!")
            return True
            
    except Exception as e:
        print(f"\n❌ Hugging Face test FAILED: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Running Universal Web Scraper Tests")
    print("=" * 60)
    
    # Test basic scraping
    basic_success = await test_basic_scraping()
    
    # Test Hugging Face scraping
    hf_success = await test_huggingface_scraping()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Basic scraping test: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Hugging Face test: {'✅ PASSED' if hf_success else '❌ FAILED'}")
    
    if basic_success and hf_success:
        print("\n🎉 All tests PASSED! The scraper is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())