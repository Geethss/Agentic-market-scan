# backend/services/rate_limiter.py
"""
Distributed & local rate limiter helpers.

- allow_domain_request(domain, max_requests, window_seconds)
    Sliding-window per-domain limiter (Redis ZSET). Returns True if allowed.

- consume_user_quota(user_id, quota_limit)
    Increment monthly quota counter and return True if allowed (i.e. not exceeded).

- ensure_local_cooldown(domain, min_interval_seconds)
    Process-local sleep to avoid bursts to the same domain.

Notes:
- Requires Redis available at cfg.REDIS_URL (backend.config.cfg).
- If Redis is unavailable, fallbacks are conservative (either allow or deny depending on function).
"""
from __future__ import annotations
import time
import logging
from typing import Optional
from redis import Redis, RedisError
from backend.config import cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_redis_client: Optional[Redis] = None

def get_redis() -> Optional[Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        _redis_client = Redis.from_url(cfg.REDIS_URL, decode_responses=True)
        # quick ping to validate connection
        _redis_client.ping()
        logger.debug("Connected to Redis for rate limiter")
        return _redis_client
    except Exception as e:
        logger.warning("Redis not available for rate limiter: %s", e)
        _redis_client = None
        return None

# -- Sliding window domain limiter (Redis ZSET) --
# Key: rl:domain:<domain>
def allow_domain_request(domain: str, max_requests: int, window_seconds: int) -> bool:
    """
    Return True if a request to 'domain' is allowed under the sliding window
    of max_requests per window_seconds. Uses Redis ZSET storing timestamps (ms).
    """
    r = get_redis()
    key = f"rl:domain:{domain}"
    now_ms = int(time.time() * 1000)
    window_start = now_ms - (window_seconds * 1000)
    try:
        if not r:
            # Redis unavailable -> be conservative: allow but log
            logger.debug("Redis unavailable; permissive fallback for domain %s", domain)
            return True

        # Remove timestamps older than the window
        r.zremrangebyscore(key, 0, window_start)
        count = r.zcard(key)
        if count >= max_requests:
            logger.debug("Domain %s blocked by distributed limiter (count=%s >= max=%s)", domain, count, max_requests)
            return False
        # Add current timestamp with score
        r.zadd(key, {str(now_ms): now_ms})
        r.expire(key, window_seconds + 5)
        return True
    except RedisError as e:
        logger.exception("Redis error in allow_domain_request: %s", e)
        # fallback allow to avoid deadlock; caller may enforce local cooldown
        return True
    except Exception as e:
        logger.exception("Unexpected error in allow_domain_request: %s", e)
        return True

# -- Monthly per-user quota (simple counter) --
# Key: quota:user:{user_id}:{YYYYMM}
def consume_user_quota(user_id: str, quota_limit: int) -> bool:
    """
    Consume 1 unit from user's monthly quota.
    Returns True if allowed (quota not exceeded), False otherwise.
    quota_limit <= 0 means unlimited.
    """
    if not user_id:
        # no user_id means anonymous — be conservative: deny heavy ops if quota >0
        logger.debug("No user_id provided to consume_user_quota; denying by default")
        return False

    if quota_limit <= 0:
        return True

    r = get_redis()
    if not r:
        # If Redis down, conservative default: deny heavy ops
        logger.warning("Redis unavailable; denying user quota consumption for safety")
        return False

    ym = time.strftime("%Y%m")
    key = f"quota:user:{user_id}:{ym}"
    try:
        new = r.incr(key)
        if new == 1:
            # Set TTL ~ 40 days so it expires around next month
            r.expire(key, 60 * 60 * 24 * 40)
        if new > quota_limit:
            logger.info("User %s exceeded monthly quota (%s > %s)", user_id, new, quota_limit)
            return False
        return True
    except RedisError as e:
        logger.exception("Redis error in consume_user_quota: %s", e)
        return False
    except Exception as e:
        logger.exception("Unexpected error in consume_user_quota: %s", e)
        return False

# -- Local in-process cooldowns --
_DOMAIN_LAST_ACCESS: dict = {}

def ensure_local_cooldown(domain: str, min_interval_seconds: float):
    """
    Simple process-local cooldown. Blocks (time.sleep) if last access to domain
    was less than min_interval_seconds ago.
    """
    try:
        last = _DOMAIN_LAST_ACCESS.get(domain)
        now = time.time()
        if last:
            delta = now - last
            if delta < min_interval_seconds:
                to_wait = min_interval_seconds - delta
                logger.debug("Local cooldown: sleeping %.3fs for domain %s", to_wait, domain)
                time.sleep(to_wait)
        _DOMAIN_LAST_ACCESS[domain] = time.time()
    except Exception as e:
        logger.debug("Error in ensure_local_cooldown: %s", e)
        return
