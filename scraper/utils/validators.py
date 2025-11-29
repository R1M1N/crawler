"""
URL validation and filtering utilities.

Provides URL validation, filtering, and URL processing functions.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from urllib.parse import urlparse, urljoin, parse_qs
from pathlib import Path


class URLValidator:
    """Validates and processes URLs."""
    
    # URL pattern for validation
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    # Suspicious patterns to filter out
    SUSPICIOUS_PATTERNS = [
        r'javascript:',
        r'data:',
        r'file:',
        r'about:',
        r'mailto:',
        r'tel:',
        r'ftp:',
        r'chrome:',
        r'chrome-extension:',
        r'moz-extension:',
        r'about:blank',
        r'about:config',
        r'view-source:'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def is_valid(self, url: str) -> bool:
        """Check if URL is valid."""
        if not url or not isinstance(url, str):
            return False
            
        url = url.strip()
        
        # Check against suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        # Check URL format
        if not self.URL_PATTERN.match(url):
            return False
            
        # Additional validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
                
            # Reject obviously invalid domains
            if not self._is_valid_domain(parsed.netloc):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain is valid."""
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
            
        # Basic domain validation
        if not domain or len(domain) > 253:
            return False
            
        # Check for valid characters
        domain_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        )
        
        return bool(domain_pattern.match(domain))
        
    def normalize_url(self, url: str, base_url: str = "") -> str:
        """Normalize URL by resolving relative URLs and cleaning up."""
        if not url:
            return ""
            
        url = url.strip()
        
        # If base_url provided and url is relative, make it absolute
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
            
        # Clean up URL
        try:
            parsed = urlparse(url)
            
            # Remove fragments
            parsed = parsed._replace(fragment='')
            
            # Sort query parameters for consistency
            if parsed.query:
                query_params = parse_qs(parsed.query)
                sorted_query = '&'.join(
                    f"{key}={value}" if not isinstance(value, list) or len(value) == 1
                    else f"{key}={'&'.join(value)}"
                    for key, value in sorted(query_params.items())
                )
                parsed = parsed._replace(query=sorted_query)
                
            # Remove trailing slashes from path (except root)
            if parsed.path.endswith('/') and len(parsed.path) > 1:
                parsed = parsed._replace(path=parsed.path.rstrip('/'))
                
            return parsed.geturl()
            
        except Exception as e:
            self.logger.warning(f"Failed to normalize URL {url}: {e}")
            return url
            
    def extract_domain(self, url: str) -> str:
        """Extract main domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
                
            return domain
            
        except Exception:
            return ""
            
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs belong to the same domain."""
        try:
            domain1 = self.extract_domain(url1)
            domain2 = self.extract_domain(url2)
            
            # For exact domain match
            if domain1 == domain2:
                return True
                
            # For subdomain matching (e.g., www.example.com vs example.com)
            if domain1.endswith('.' + domain2) or domain2.endswith('.' + domain1):
                return True
                
            return False
            
        except Exception:
            return False
            
    def is_subdomain(self, url: str, main_domain: str) -> bool:
        """Check if URL is a subdomain of the main domain."""
        try:
            domain = self.extract_domain(url)
            main_domain = main_domain.lower()
            
            # Remove www if present for comparison
            if domain.startswith('www.'):
                domain = domain[4:]
            if main_domain.startswith('www.'):
                main_domain = main_domain[4:]
                
            # Check if domain is subdomain of main domain
            return domain != main_domain and (domain.endswith('.' + main_domain) or main_domain.endswith('.' + domain))
            
        except Exception:
            return False
            
    def get_url_depth(self, url: str) -> int:
        """Get the depth of URL (number of path segments)."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            if not path:
                return 0
                
            return len([seg for seg in path.split('/') if seg])
            
        except Exception:
            return 0
            
    def clean_url_for_storage(self, url: str) -> str:
        """Clean URL for safe storage and display."""
        try:
            # Remove common tracking parameters
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            # Remove tracking parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'mc_cid', 'mc_eid', '_ga', '_gid', '_gac',
                'ref', 'referrer', 'source', 'campaign', 'medium'
            }
            
            cleaned_params = {
                key: value for key, value in query_params.items()
                if key.lower() not in tracking_params
            }
            
            # Rebuild URL
            if cleaned_params:
                query = '&'.join(
                    f"{key}={value}" if not isinstance(value, list) or len(value) == 1
                    else f"{key}={'&'.join(value)}"
                    for key, value in sorted(cleaned_params.items())
                )
                parsed = parsed._replace(query=query)
                
            return parsed.geturl()
            
        except Exception:
            return url


class URLFilter:
    """Filters and processes URLs based on various criteria."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if URL matches a pattern (simple wildcard matching)."""
        try:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace('*', '.*').replace('?', '.')
            return bool(re.match(f'^{regex_pattern}$', url, re.IGNORECASE))
        except Exception:
            return False
            
    def matches_any_pattern(self, url: str, patterns: List[str]) -> bool:
        """Check if URL matches any of the provided patterns."""
        if not patterns:
            return True
            
        for pattern in patterns:
            if self.matches_pattern(url, pattern):
                return True
        return False
        
    def is_robots_allowed(self, url: str, user_agent: str = '*') -> bool:
        """Check if URL is allowed by robots.txt (simplified implementation)."""
        try:
            from urllib.robotparser import RobotFileParser
            
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(user_agent, url)
            
        except Exception:
            # If robots.txt check fails, allow the URL
            return True
            
    def filter_urls(self, 
                   urls: List[str],
                   allowed_patterns: Optional[List[str]] = None,
                   exclude_patterns: Optional[List[str]] = None,
                   allowed_extensions: Optional[List[str]] = None,
                   exclude_extensions: Optional[List[str]] = None,
                   max_depth: Optional[int] = None,
                   stay_on_domain: bool = False,
                   base_domain: Optional[str] = None,
                   respect_robots: bool = False,
                   user_agent: str = '*') -> List[str]:
        """
        Filter URLs based on multiple criteria.
        
        Args:
            urls: List of URLs to filter
            allowed_patterns: Patterns to include (regex or glob)
            exclude_patterns: Patterns to exclude (regex or glob)
            allowed_extensions: File extensions to include
            exclude_extensions: File extensions to exclude
            max_depth: Maximum URL depth
            stay_on_domain: Whether to stay on the same domain
            base_domain: Base domain for comparison
            respect_robots: Whether to check robots.txt
            user_agent: User agent for robots.txt check
            
        Returns:
            Filtered list of URLs
        """
        validator = URLValidator()
        filtered_urls = []
        
        for url in urls:
            try:
                # Basic validation
                if not validator.is_valid(url):
                    continue
                    
                # Robots.txt check
                if respect_robots and not self.is_robots_allowed(url, user_agent):
                    continue
                    
                # Pattern filtering
                if exclude_patterns and self.matches_any_pattern(url, exclude_patterns):
                    continue
                    
                if allowed_patterns and not self.matches_any_pattern(url, allowed_patterns):
                    continue
                    
                # Extension filtering
                parsed_url = urlparse(url)
                path = parsed_url.path.lower()
                
                if exclude_extensions and any(path.endswith(ext) for ext in exclude_extensions):
                    continue
                    
                if allowed_extensions and not any(path.endswith(ext) for ext in allowed_extensions):
                    continue
                    
                # Depth filtering
                if max_depth is not None and validator.get_url_depth(url) > max_depth:
                    continue
                    
                # Domain filtering
                if stay_on_domain and base_domain and not validator.is_same_domain(url, base_domain):
                    continue
                    
                # All checks passed
                filtered_urls.append(url)
                
            except Exception as e:
                self.logger.warning(f"Failed to filter URL {url}: {e}")
                continue
                
        return filtered_urls
        
    def deduplicate_urls(self, urls: List[str], keep_first: bool = True) -> List[str]:
        """Remove duplicate URLs while preserving order."""
        seen = set()
        unique_urls = []
        
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
            elif not keep_first:
                # If not keeping first, skip this URL
                continue
                
        return unique_urls
        
    def prioritize_urls(self, urls: List[str], preferences: Dict[str, int]) -> List[str]:
        """
        Prioritize URLs based on preferences.
        
        Args:
            urls: List of URLs to prioritize
            preferences: Dictionary mapping patterns to priority scores (higher = more important)
            
        Returns:
            URLs sorted by priority
        """
        def get_priority(url: str) -> int:
            max_priority = 0
            for pattern, priority in preferences.items():
                if self.matches_pattern(url, pattern):
                    max_priority = max(max_priority, priority)
            return max_priority
            
        return sorted(urls, key=get_priority, reverse=True)
        
    def categorize_urls(self, urls: List[str]) -> Dict[str, List[str]]:
        """Categorize URLs by type."""
        categories = {
            'html': [],
            'images': [],
            'documents': [],
            'audio': [],
            'video': [],
            'api': [],
            'other': []
        }
        
        for url in urls:
            try:
                parsed_url = urlparse(url)
                path = parsed_url.path.lower()
                
                # HTML/web pages
                if any(path.endswith(ext) for ext in ['.html', '.htm', '.php', '.asp', '.aspx', '.jsp']):
                    categories['html'].append(url)
                    
                # Images
                elif any(path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff']):
                    categories['images'].append(url)
                    
                # Documents
                elif any(path.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt']):
                    categories['documents'].append(url)
                    
                # Audio
                elif any(path.endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a']):
                    categories['audio'].append(url)
                    
                # Video
                elif any(path.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']):
                    categories['video'].append(url)
                    
                # API endpoints
                elif 'api' in path or 'rest' in path or parsed_url.netloc.endswith('.api.'):
                    categories['api'].append(url)
                    
                else:
                    categories['other'].append(url)
                    
            except Exception:
                categories['other'].append(url)
                
        return categories
        
    def extract_url_parameters(self, url: str) -> Dict[str, Any]:
        """Extract and categorize URL parameters."""
        try:
            parsed_url = urlparse(url)
            params = parse_qs(parsed_url.query)
            
            categorized_params = {
                'tracking': {},
                'navigation': {},
                'content': {},
                'other': {}
            }
            
            # Categorize parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'mc_cid', 'mc_eid', '_ga', '_gid', '_gac',
                'ref', 'referrer', 'source', 'campaign', 'medium'
            }
            
            navigation_params = {
                'page', 'p', 'offset', 'limit', 'start', 'skip', 'cursor'
            }
            
            for key, values in params.items():
                key_lower = key.lower()
                
                if key_lower in tracking_params:
                    categorized_params['tracking'][key] = values
                elif key_lower in navigation_params:
                    categorized_params['navigation'][key] = values
                elif key in ['id', 'slug', 'title', 'content', 'query']:
                    categorized_params['content'][key] = values
                else:
                    categorized_params['other'][key] = values
                    
            return categorized_params
            
        except Exception:
            return {'tracking': {}, 'navigation': {}, 'content': {}, 'other': {}}
            
    def get_url_score(self, url: str, factors: Dict[str, float]) -> float:
        """
        Calculate URL score based on various factors.
        
        Args:
            url: URL to score
            factors: Dictionary of factor weights
            
        Returns:
            Score between 0 and 1
        """
        try:
            parsed_url = urlparse(url)
            score = 0.0
            total_weight = sum(factors.values())
            
            # URL length factor
            if 'url_length' in factors:
                length_score = min(len(url) / 100, 1.0)  # Normalize to 0-1
                score += factors['url_length'] * length_score
                
            # Depth factor
            if 'depth' in factors:
                depth = len([seg for seg in parsed_url.path.split('/') if seg])
                depth_score = max(0, 1 - (depth / 5))  # Prefer shallower URLs
                score += factors['depth'] * depth_score
                
            # Parameter count factor
            if 'param_count' in factors:
                param_count = len(parse_qs(parsed_url.query))
                param_score = max(0, 1 - (param_count / 10))  # Prefer fewer parameters
                score += factors['param_count'] * param_score
                
            # Extension factor
            if 'extension' in factors:
                path = parsed_url.path.lower()
                if path.endswith(('.html', '.htm')):
                    score += factors['extension'] * 1.0
                elif path.endswith(('.pdf', '.doc', '.docx')):
                    score += factors['extension'] * 0.8
                else:
                    score += factors['extension'] * 0.5
                    
            return min(score / total_weight, 1.0) if total_weight > 0 else 0.0
            
        except Exception:
            return 0.0