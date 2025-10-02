# backend/services/url_cache.py
"""
URL content cache service using Redis.

Caches fetched full_text content to avoid re-fetching the same URLs across jobs.
Uses Redis with configurable TTL and compression for storage efficiency.

Features:
- Cache full_text content with metadata (title, published date)
- Configurable TTL (default 7 days)
- Compression for large content
- Cache hit/miss statistics
- Graceful fallback when Redis unavailable
"""
from __future__ import annotations
import json
import gzip
import base64
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from backend.config import cfg
from backend.services.rate_limiter import get_redis

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Cache configuration
CACHE_TTL_DAYS = 7  # How long to cache content
CACHE_PREFIX = "url_cache:"  # Redis key prefix
COMPRESS_THRESHOLD = 1024  # Compress content larger than 1KB

# Cache statistics
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "errors": 0
}


def _get_cache_key(url: str) -> str:
    """Generate a consistent cache key for a URL."""
    # Use URL hash for consistent key generation
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"{CACHE_PREFIX}{url_hash}"


def _compress_content(content: str) -> str:
    """Compress content if it's large enough."""
    if len(content) < COMPRESS_THRESHOLD:
        return content
    
    try:
        compressed = gzip.compress(content.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')
        return f"gzip:{encoded}"
    except Exception as e:
        logger.warning("Failed to compress content: %s", e)
        return content


def _decompress_content(content: str) -> str:
    """Decompress content if it's compressed."""
    if not content.startswith("gzip:"):
        return content
    
    try:
        encoded = content[5:]  # Remove "gzip:" prefix
        compressed = base64.b64decode(encoded)
        decompressed = gzip.decompress(compressed)
        return decompressed.decode('utf-8')
    except Exception as e:
        logger.warning("Failed to decompress content: %s", e)
        return content


def get_cached_content(url: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached content for a URL.
    
    Returns:
        Dict with keys: title, text, published, cached_at, compressed
        None if not cached or error
    """
    global _cache_stats
    
    redis_client = get_redis()
    if not redis_client:
        _cache_stats["errors"] += 1
        return None
    
    try:
        cache_key = _get_cache_key(url)
        cached_data = redis_client.get(cache_key)
        
        if not cached_data:
            _cache_stats["misses"] += 1
            return None
        
        # Parse cached data
        data = json.loads(cached_data)
        
        # Decompress text if needed
        if data.get("compressed", False):
            data["text"] = _decompress_content(data["text"])
        
        _cache_stats["hits"] += 1
        logger.debug("Cache hit for URL: %s", url)
        return data
        
    except Exception as e:
        logger.exception("Error retrieving cached content for %s: %s", url, e)
        _cache_stats["errors"] += 1
        return None


def set_cached_content(url: str, title: str, text: str, published: Optional[str] = None) -> bool:
    """
    Cache content for a URL.
    
    Args:
        url: The URL that was fetched
        title: Page title
        text: Full text content
        published: Publication date (ISO format)
    
    Returns:
        True if successfully cached, False otherwise
    """
    global _cache_stats
    
    redis_client = get_redis()
    if not redis_client:
        _cache_stats["errors"] += 1
        return False
    
    try:
        # Compress text if it's large
        original_text = text
        compressed = len(text) >= COMPRESS_THRESHOLD
        if compressed:
            text = _compress_content(text)
        
        # Prepare cache data
        cache_data = {
            "url": url,
            "title": title,
            "text": text,
            "published": published,
            "cached_at": datetime.utcnow().isoformat(),
            "compressed": compressed,
            "original_size": len(original_text),
            "cached_size": len(text)
        }
        
        # Store in Redis with TTL
        cache_key = _get_cache_key(url)
        ttl_seconds = int(timedelta(days=CACHE_TTL_DAYS).total_seconds())
        
        redis_client.setex(
            cache_key,
            ttl_seconds,
            json.dumps(cache_data)
        )
        
        logger.debug("Cached content for URL: %s (compressed: %s, size: %d -> %d)", 
                    url, compressed, len(original_text), len(text))
        return True
        
    except Exception as e:
        logger.exception("Error caching content for %s: %s", url, e)
        _cache_stats["errors"] += 1
        return False


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    total_requests = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = (_cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "errors": _cache_stats["errors"],
        "total_requests": total_requests,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_ttl_days": CACHE_TTL_DAYS
    }


def clear_cache_stats() -> None:
    """Reset cache statistics."""
    global _cache_stats
    _cache_stats = {"hits": 0, "misses": 0, "errors": 0}


def clear_url_cache(url: str) -> bool:
    """Remove a specific URL from cache."""
    redis_client = get_redis()
    if not redis_client:
        return False
    
    try:
        cache_key = _get_cache_key(url)
        result = redis_client.delete(cache_key)
        logger.debug("Cleared cache for URL: %s (deleted: %s)", url, bool(result))
        return bool(result)
    except Exception as e:
        logger.exception("Error clearing cache for %s: %s", url, e)
        return False


def clear_all_cache() -> bool:
    """Clear all cached URLs (use with caution)."""
    redis_client = get_redis()
    if not redis_client:
        return False
    
    try:
        pattern = f"{CACHE_PREFIX}*"
        keys = redis_client.keys(pattern)
        if keys:
            deleted = redis_client.delete(*keys)
            logger.info("Cleared %d cached URLs", deleted)
            return True
        return True
    except Exception as e:
        logger.exception("Error clearing all cache: %s", e)
        return False


# CLI for cache management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="URL Cache Management")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear-url", help="Clear cache for specific URL")
    parser.add_argument("--clear-all", action="store_true", help="Clear all cached URLs")
    parser.add_argument("--reset-stats", action="store_true", help="Reset cache statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_cache_stats()
        print(json.dumps(stats, indent=2))
    elif args.clear_url:
        success = clear_url_cache(args.clear_url)
        print(f"Cleared cache for {args.clear_url}: {success}")
    elif args.clear_all:
        success = clear_all_cache()
        print(f"Cleared all cache: {success}")
    elif args.reset_stats:
        clear_cache_stats()
        print("Cache statistics reset")
    else:
        parser.print_help()
