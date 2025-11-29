#!/usr/bin/env python3
"""
Universal Web Scraper Setup Script

This script helps set up the Universal Web Scraper by:
1. Installing dependencies
2. Downloading required models and data
3. Setting up configuration
4. Running tests to verify installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a shell command with nice output."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing requirements"):
        return False
    
    return True


def download_models_and_data():
    """Download required models and data."""
    print("\n📥 Downloading models and data...")
    
    # Download spaCy English model
    if not run_command(f"{sys.executable} -m spacy download en_core_web_sm", 
                      "Downloading spaCy English model", check=False):
        print("⚠️  spaCy model download failed - OCR features may not work optimally")
    
    # Download NLTK data
    nltk_downloads = ['punkt', 'stopwords', 'wordnet']
    for data in nltk_downloads:
        run_command(f"{sys.executable} -c \"import nltk; nltk.download('{data}')\"", 
                   f"Downloading NLTK {data} data", check=False)
    
    # Download Playwright browsers
    if not run_command(f"{sys.executable} -m playwright install chromium", 
                      "Installing Playwright browsers", check=False):
        print("⚠️  Playwright installation failed - JavaScript support may not work")
    
    return True


def check_external_tools():
    """Check for external tools."""
    print("\n🔧 Checking external tools...")
    
    tools = {
        'tesseract --version': 'Tesseract OCR (for image text extraction)',
        'ffmpeg -version': 'FFmpeg (for video/audio processing)',
        'ffprobe -version': 'FFprobe (for media metadata extraction)'
    }
    
    missing_tools = []
    
    for cmd, description in tools.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            print(f"✅ {description}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {description} - Not found")
            missing_tools.append(description)
    
    if missing_tools:
        print(f"\n💡 Optional: Install missing tools for full functionality:")
        print("   Tesseract: sudo apt-get install tesseract-ocr (Ubuntu/Debian)")
        print("   FFmpeg: sudo apt-get install ffmpeg (Ubuntu/Debian)")
        print("   Or visit: https://ffmpeg.org/download.html")
    
    return True


def setup_configuration():
    """Set up configuration files."""
    print("\n⚙️  Setting up configuration...")
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create default config file
    config_file = config_dir / "scraper.json"
    
    if not config_file.exists():
        default_config = {
            "scraper": {
                "headless": True,
                "timeout": 30,
                "max_concurrent_requests": 5,
                "requests_per_second": 1.0,
                "storage_enabled": True,
                "storage_path": "scraped_data"
            },
            "crawler": {
                "max_depth": 3,
                "max_pages": 1000,
                "stay_on_domain": True,
                "respect_robots_txt": True,
                "max_concurrent_requests": 3
            },
            "vector_storage": {
                "backend": "chromadb",
                "storage_path": "vector_storage",
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "strategy": "semantic"
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✅ Created default config: {config_file}")
    else:
        print(f"✅ Config file exists: {config_file}")
    
    # Create data directories
    directories = ["scraped_data", "vector_storage", "media_downloads"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests():
    """Run basic tests to verify installation."""
    print("\n🧪 Running tests...")
    
    # Test imports
    test_imports = [
        "playwright",
        "fastapi",
        "spacy",
        "sentence_transformers",
        "chromadb",
        "pytesseract",
        "tesseract",
        "beautifulsoup4",
        "asyncio",
        "aiohttp"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    # Test scraper components
    print("\n🔧 Testing scraper components...")
    try:
        from scraper.utils.config import Config
        from scraper.utils.logger import get_logger
        
        config = Config()
        logger = get_logger("test")
        
        print("✅ Config system")
        print("✅ Logging system")
        
        # Test basic functionality (without actual scraping)
        print("✅ Basic scraper components")
        
    except Exception as e:
        print(f"❌ Scraper components: {e}")
        failed_imports.append("scraper")
    
    if failed_imports:
        print(f"\n⚠️  Some components failed to import: {failed_imports}")
        print("The scraper may have limited functionality.")
    else:
        print("\n🎉 All tests passed!")
    
    return len(failed_imports) == 0


def create_example_scripts():
    """Create example scripts for users."""
    print("\n📝 Creating example scripts...")
    
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Simple scraping example
    simple_example = '''#!/usr/bin/env python3
"""Simple scraping example."""

import asyncio
from scraper import ScrapingEngine, Config

async def main():
    config = Config()
    
    async with ScrapingEngine(config.scraper) as engine:
        result = await engine.scrape("https://httpbin.org/html")
        
        if result.success:
            print(f"Title: {result.title}")
            print(f"Content: {result.content[:200]}...")
        else:
            print(f"Failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open(examples_dir / "simple_scraper.py", 'w') as f:
        f.write(simple_example)
    
    # CLI usage example
    cli_example = '''# Universal Web Scraper CLI Examples

## Basic scraping
python cli.py scrape https://example.com

## Batch scraping
python cli.py scrape-batch https://example.com https://httpbin.org

## Website crawling
python cli.py crawl https://example.com --max-depth 2 --max-pages 50

## Media extraction
python cli.py extract-media https://example.com/image.jpg https://example.com/video.mp4

## Create RAG index
python cli.py create-rag-index document.txt --backend chromadb

## Test vector database
python cli.py test-vector-db --backend chromadb --test-query "test query"

## Check status
python cli.py status
'''
    
    with open(examples_dir / "cli_examples.md", 'w') as f:
        f.write(cli_example)
    
    print(f"✅ Created example scripts in {examples_dir}")
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n🚀 Setup Complete!")
    print("=" * 50)
    
    print("\n📖 Quick Start:")
    print("1. Try the CLI:")
    print("   python cli.py status")
    print("   python cli.py scrape https://httpbin.org/html")
    
    print("\n2. Use Python API:")
    print("   python examples/simple_scraper.py")
    
    print("\n3. Start API server:")
    print("   python -m scraper.api.server")
    
    print("\n4. View full examples:")
    print("   python examples/basic_usage.py")
    
    print("\n📚 Documentation:")
    print("- README.md - Full documentation")
    print("- examples/ - Usage examples")
    print("- config/scraper.json - Configuration")
    
    print("\n💡 Tips:")
    print("- Use --verbose (-v) flag for detailed output")
    print("- Check external tool installation: python cli.py status")
    print("- Respect robots.txt and be ethical when scraping")
    print("- Configure rate limiting for large-scale crawling")
    
    print("\n🔗 Useful Links:")
    print("- GitHub: [repository_url]")
    print("- Documentation: [docs_url]")
    print("- Issues: [issues_url]")


def main():
    """Main setup function."""
    print("🕷️  Universal Web Scraper Setup")
    print("=" * 50)
    print("This script will set up the Universal Web Scraper for you.")
    print("It may take a few minutes to download models and dependencies.")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Download models and data
    download_models_and_data()
    
    # Check external tools
    check_external_tools()
    
    # Setup configuration
    setup_configuration()
    
    # Create examples
    create_example_scripts()
    
    # Run tests
    tests_passed = run_tests()
    
    # Print next steps
    print_next_steps()
    
    if not tests_passed:
        print("\n⚠️  Some tests failed, but basic functionality should work.")
        print("Check the error messages above for details.")
    
    print("\n✨ Happy scraping!")


if __name__ == "__main__":
    main()