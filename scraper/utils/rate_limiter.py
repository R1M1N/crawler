"""
Rate limiting utilities.

Provides rate limiting mechanisms for web scraping operations.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from collections import deque
import logging


class RateLimiter:
    """Rate limiter for controlling request frequency."""
    
    def __init__(self, 
                 requests_per_second: float = 1.0,
                 delay_between_requests: float = 1.0,
                 max_burst: int = 5):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            delay_between_requests: Minimum delay between requests in seconds
            max_burst: Maximum burst size (concurrent requests allowed)
        """
        self.requests_per_second = requests_per_second
        self.delay_between_requests = delay_between_requests
        self.max_burst = max_burst
        
        # For token bucket algorithm
        self.tokens = max_burst
        self.max_tokens = max_burst
        self.last_update = time.time()
        
        # For sliding window
        self.request_times = deque()
        
        self.logger = logging.getLogger(__name__)
        
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens for making requests.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        # Try token bucket first
        if self._acquire_tokens(tokens):
            return True
            
        # Fall back to sliding window
        return await self._wait_for_slot()
        
    def _acquire_tokens(self, tokens: int) -> bool:
        """Try to acquire tokens using token bucket algorithm."""
        now = time.time()
        
        # Add tokens based on time passed
        time_passed = now - self.last_update
        self.tokens = min(
            self.max_tokens,
            self.tokens + time_passed * self.requests_per_second
        )
        self.last_update = now
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
            
        return False
        
    async def _wait_for_slot(self) -> bool:
        """Wait for a slot using sliding window rate limiting."""
        now = time.time()
        
        # Remove old requests from sliding window
        window_start = now - 1.0  # 1 second window
        while self.request_times and self.request_times[0] < window_start:
            self.request_times.popleft()
            
        # Check if we're under the rate limit
        if len(self.request_times) < self.max_burst:
            self.request_times.append(now)
            return True
            
        # Wait for the oldest request to expire
        oldest_time = self.request_times[0]
        wait_time = max(0, 1.0 - (now - oldest_time) + 0.001)  # Small buffer
        
        self.logger.debug(f"Rate limited, waiting {wait_time:.3f} seconds")
        await asyncio.sleep(wait_time)
        
        # Try again
        return await self.acquire()
        
    def can_make_request(self, tokens: int = 1) -> bool:
        """Check if a request can be made without waiting."""
        return self._acquire_tokens(tokens)
        
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for acquiring tokens."""
        if self._acquire_tokens(tokens):
            # We got tokens immediately, return 0
            return 0.0
            
        # Calculate wait time based on sliding window
        now = time.time()
        window_start = now - 1.0
        
        # Remove old requests
        current_requests = [
            t for t in self.request_times if t > window_start
        ]
        
        if len(current_requests) < self.max_burst:
            return 0.0
            
        # Calculate wait time for oldest request
        oldest_time = current_requests[0]
        return max(0, 1.0 - (now - oldest_time))
        
    def reset(self):
        """Reset the rate limiter state."""
        self.tokens = self.max_tokens
        self.last_update = time.time()
        self.request_times.clear()
        
    def update_rate(self, requests_per_second: float, max_burst: Optional[int] = None):
        """Update rate limiting parameters."""
        old_max_tokens = self.max_tokens
        
        self.requests_per_second = requests_per_second
        if max_burst is not None:
            self.max_burst = max_burst
            self.max_tokens = max_burst
            self.tokens = min(self.tokens, self.max_tokens)
        elif max_burst is None and self.max_tokens != self.max_burst:
            # If max_burst changed, update tokens proportionally
            ratio = self.max_tokens / old_max_tokens
            self.tokens = min(self.tokens * ratio, self.max_tokens)


class DomainRateLimiter:
    """Rate limiter that tracks rates per domain."""
    
    def __init__(self, 
                 default_requests_per_second: float = 1.0,
                 default_max_burst: int = 3):
        """
        Initialize domain-specific rate limiter.
        
        Args:
            default_requests_per_second: Default rate limit
            default_max_burst: Default burst limit
        """
        self.default_rps = default_requests_per_second
        self.default_max_burst = default_max_burst
        
        # Domain-specific rate limiters
        self.limiters: Dict[str, RateLimiter] = {}
        
        self.logger = logging.getLogger(__name__)
        
    def get_limiter(self, domain: str) -> RateLimiter:
        """Get or create a rate limiter for a domain."""
        if domain not in self.limiters:
            self.limiters[domain] = RateLimiter(
                requests_per_second=self.default_rps,
                delay_between_requests=1.0 / self.default_rps,
                max_burst=self.default_max_burst
            )
        return self.limiters[domain]
        
    async def acquire(self, domain: str, tokens: int = 1) -> bool:
        """Acquire tokens for a specific domain."""
        limiter = self.get_limiter(domain)
        return await limiter.acquire(tokens)
        
    def set_domain_rate(self, 
                       domain: str, 
                       requests_per_second: float,
                       max_burst: Optional[int] = None):
        """Set custom rate limits for a domain."""
        limiter = self.get_limiter(domain)
        limiter.update_rate(requests_per_second, max_burst)
        
    def can_make_request(self, domain: str, tokens: int = 1) -> bool:
        """Check if request can be made without waiting."""
        limiter = self.get_limiter(domain)
        return limiter.can_make_request(tokens)
        
    def get_wait_time(self, domain: str, tokens: int = 1) -> float:
        """Get wait time for a domain."""
        limiter = self.get_limiter(domain)
        return limiter.get_wait_time(tokens)
        
    def reset_domain(self, domain: str):
        """Reset rate limiter for a domain."""
        if domain in self.limiters:
            self.limiters[domain].reset()
            
    def reset_all(self):
        """Reset all domain limiters."""
        for limiter in self.limiters.values():
            limiter.reset()
            
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain."""
        limiter = self.get_limiter(domain)
        return {
            'requests_per_second': limiter.requests_per_second,
            'max_burst': limiter.max_burst,
            'current_tokens': limiter.tokens,
            'current_requests_in_window': len(limiter.request_times)
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on response times and errors."""
    
    def __init__(self,
                 initial_rps: float = 1.0,
                 min_rps: float = 0.1,
                 max_rps: float = 10.0,
                 error_threshold: float = 0.1,
                 success_threshold: float = 0.9):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rps: Initial requests per second
            min_rps: Minimum rate limit
            max_rps: Maximum rate limit
            error_threshold: Error rate threshold for slowing down
            success_threshold: Success rate threshold for speeding up
        """
        self.current_rps = initial_rps
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.error_threshold = error_threshold
        self.success_threshold = success_threshold
        
        # Statistics tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        
        # Current rate limiter
        self.limiter = RateLimiter(
            requests_per_second=initial_rps,
            delay_between_requests=1.0 / initial_rps,
            max_burst=int(initial_rps * 2)  # Allow some burst
        )
        
        self.logger = logging.getLogger(__name__)
        
    async def acquire(self) -> bool:
        """Acquire tokens for making a request."""
        return await self.limiter.acquire()
        
    def record_success(self, response_time: float):
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        
        self._adapt_rate()
        
    def record_failure(self, response_time: Optional[float] = None):
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        
        if response_time is not None:
            self.total_response_time += response_time
            
        self._adapt_rate()
        
    def _adapt_rate(self):
        """Adapt the rate based on recent performance."""
        if self.total_requests < 10:  # Need enough samples
            return
            
        success_rate = self.successful_requests / self.total_requests
        avg_response_time = self.total_response_time / self.total_requests
        
        # Adjust rate based on success rate
        if success_rate < self.error_threshold:
            # Too many errors, slow down
            self.current_rps = max(self.min_rps, self.current_rps * 0.8)
            self.logger.info(f"High error rate ({success_rate:.2%}), reducing rate to {self.current_rps:.2f} RPS")
            
        elif success_rate > self.success_threshold and avg_response_time < 2.0:
            # Low error rate and fast responses, speed up
            self.current_rps = min(self.max_rps, self.current_rps * 1.1)
            self.logger.debug(f"Low error rate ({success_rate:.2%}), increasing rate to {self.current_rps:.2f} RPS")
            
        # Update the underlying rate limiter
        self.limiter.update_rate(self.current_rps)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'current_rps': self.current_rps,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(1, self.total_requests),
            'avg_response_time': self.total_response_time / max(1, self.total_requests),
            'max_burst': self.limiter.max_burst,
            'current_tokens': self.limiter.tokens
        }
        
    def reset(self):
        """Reset statistics and rate."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.current_rps = max(self.min_rps, self.current_rps * 0.5)
        self.limiter.update_rate(self.current_rps)
        self.limiter.reset()