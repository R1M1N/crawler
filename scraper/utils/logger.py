"""
Logging utility for the Universal Web Scraper.

Provides centralized logging configuration and utility functions.
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            
        # Add color to module name
        if hasattr(record, 'module'):
            record.module = f"\033[90m{record.module}\033[0m"  # Dark gray
            
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
                
        return json.dumps(log_entry, default=str)


class ScrapingFilter(logging.Filter):
    """Custom filter for adding context to log messages."""
    
    def __init__(self, context=None):
        super().__init__()
        self.context = context or {}
        
    def filter(self, record):
        # Add context to the record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class ScraperLogger:
    """Centralized logging configuration for the scraper."""
    
    _loggers = {}
    
    def __init__(self, 
                 name: str,
                 level: str = "INFO",
                 log_file: Optional[Path] = None,
                 console: bool = True,
                 json_format: bool = False,
                 context: Optional[dict] = None):
        self.name = name
        self.level = level.upper()
        self.log_file = log_file
        self.console = console
        self.json_format = json_format
        self.context = context or {}
        
    def get_logger(self) -> logging.Logger:
        """Get or create a logger instance."""
        if self.name in self._loggers:
            return self._loggers[self.name]
            
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        if self.json_format:
            formatter = JSONFormatter()
        else:
            formatter = ColoredFormatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
        # File handler
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        # Console handler
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        # Add context filter
        if self.context:
            context_filter = ScrapingFilter(self.context)
            logger.addFilter(context_filter)
            
        self._loggers[self.name] = logger
        return logger


# Predefined logger configurations
def get_logger(name: str = "scraper", 
               level: str = "INFO",
               log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Get a configured logger for the scraper.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    scraper_logger = ScraperLogger(
        name=name,
        level=level,
        log_file=log_dir / f"{name}.log" if log_dir else None,
        console=True,
        json_format=False
    )
    
    return scraper_logger.get_logger()


def get_scraping_logger(url: str = None, job_id: str = None) -> logging.Logger:
    """
    Get a logger with scraping-specific context.
    
    Args:
        url: URL being scraped
        job_id: Job identifier
        
    Returns:
        Configured logger with context
    """
    context = {}
    if url:
        context['scraping_url'] = url
    if job_id:
        context['job_id'] = job_id
        
    scraper_logger = ScraperLogger(
        name="scraper",
        context=context,
        console=True,
        json_format=False
    )
    
    return scraper_logger.get_logger()


def get_crawler_logger(domain: str = None, crawl_id: str = None) -> logging.Logger:
    """
    Get a logger with crawler-specific context.
    
    Args:
        domain: Domain being crawled
        crawl_id: Crawl job identifier
        
    Returns:
        Configured logger with context
    """
    context = {}
    if domain:
        context['crawl_domain'] = domain
    if crawl_id:
        context['crawl_id'] = crawl_id
        
    scraper_logger = ScraperLogger(
        name="crawler",
        context=context,
        console=True,
        json_format=False
    )
    
    return scraper_logger.get_logger()


def get_api_logger() -> logging.Logger:
    """Get a logger for API operations."""
    return get_logger("api")


def get_worker_logger() -> logging.Logger:
    """Get a logger for background workers."""
    return get_logger("worker")


def setup_logging(log_level: str = "INFO",
                  log_dir: Optional[Path] = None,
                  structured_logging: bool = False) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_level: Global logging level
        log_dir: Directory for log files
        structured_logging: Use JSON format for logs
    """
    # Create log directory
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if structured_logging:
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_file = log_dir / "scraper.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    # Set third-party loggers to warning level to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logger = get_logger("setup")
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file if log_dir else 'Console only'}")


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_scraping_performance(self, 
                                url: str,
                                duration: float,
                                success: bool,
                                content_size: int = 0,
                                page_count: int = 0):
        """Log scraping performance metrics."""
        self.logger.info(
            f"Scraping performance | URL: {url} | Duration: {duration:.2f}s | "
            f"Success: {success} | Content: {content_size} bytes | Pages: {page_count}",
            extra={
                'event_type': 'scraping_performance',
                'url': url,
                'duration': duration,
                'success': success,
                'content_size': content_size,
                'page_count': page_count
            }
        )
        
    def log_crawling_performance(self,
                                domain: str,
                                urls_discovered: int,
                                urls_crawled: int,
                                urls_successful: int,
                                total_duration: float):
        """Log crawling performance metrics."""
        success_rate = (urls_successful / urls_crawled * 100) if urls_crawled > 0 else 0
        
        self.logger.info(
            f"Crawling performance | Domain: {domain} | Discovered: {urls_discovered} | "
            f"Crawled: {urls_crawled} | Success: {urls_successful} | "
            f"Success Rate: {success_rate:.1f}% | Duration: {total_duration:.2f}s",
            extra={
                'event_type': 'crawling_performance',
                'domain': domain,
                'urls_discovered': urls_discovered,
                'urls_crawled': urls_crawled,
                'urls_successful': urls_successful,
                'success_rate': success_rate,
                'total_duration': total_duration
            }
        )
        
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage."""
        self.logger.debug(
            f"Memory usage | Component: {component} | Memory: {memory_mb:.1f}MB",
            extra={
                'event_type': 'memory_usage',
                'component': component,
                'memory_mb': memory_mb
            }
        )


def create_performance_logger(name: str = "performance") -> PerformanceLogger:
    """Create a performance logger instance."""
    logger = get_logger(name)
    return PerformanceLogger(logger)


# Context manager for timing operations
class LoggingTimer:
    """Context manager for timing operations with logging."""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation}", extra=self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.debug(
                f"Completed operation: {self.operation} in {duration:.2f}s",
                extra={**self.context, 'duration': duration}
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation} after {duration:.2f}s - {exc_val}",
                extra={**self.context, 'duration': duration, 'error': str(exc_val)}
            )


def time_operation(logger: logging.Logger, operation: str, **context):
    """Decorator for timing function operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LoggingTimer(logger, operation, **context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Utility function for log levels
def parse_log_level(level_str: str) -> int:
    """Parse log level string to logging constant."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    return level_map.get(level_str.upper(), logging.INFO)