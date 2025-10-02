# backend/agents/scout.py
"""
Scout agent (production-ready) with strict rate limiting integration.

- Uses Tavily Search API for relevance (requires TAVILY_API_KEY).
- Optionally fetches full HTML text for Premium users (fetch_full=True).
- Enforces per-domain rate limits and per-user monthly full-fetch quotas (via Redis).
- Uses async httpx for concurrent fetches, but exposes synchronous `scout()` for pipeline use.

scout(subqueries, max_urls_per_query=5, fetch_full=False, user_id=None, user_plan="free")
"""
from __future__ import annotations
import os
import uuid
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from backend.config import cfg
from backend.services.rate_limiter import allow_domain_request, ensure_local_cooldown, consume_user_quota
from backend.services.url_cache import get_cached_content, set_cached_content

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TAVILY_API_KEY = getattr(cfg, "TAVILY_API_KEY", None) or os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# HTTP / concurrency settings
HTTP_TIMEOUT = 15.0
MAX_CONCURRENT_FETCHES = 6
FETCH_RETRIES = 2
USER_AGENT = "AgenticMarketScan/1.0 (+https://yourdomain.example) " \
             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) " \
             "Chrome/117.0.0.0 Safari/537.36"

# Rate limit config (tune these)
PER_DOMAIN_MAX = 5          # max requests per domain
PER_DOMAIN_WINDOW = 60      # window in seconds
PER_DOMAIN_MIN_INTERVAL = 0.6  # local min interval between requests to same domain (seconds)

# Monthly quotas per plan for full fetches (change as appropriate)
PLAN_QUOTAS = {
    "free": 10,     # 10 full fetches / month
    "pro": 100,     # 100 full fetches / month
    "premium": -1   # -1 means unlimited
}


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _is_safe_scheme(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https")


def _extract_text_from_html(html: str, base_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # attempt to find JSON-LD script with datePublished
    published = None
    try:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict) and d.get("datePublished"):
                            published = d.get("datePublished")
                            break
                elif isinstance(data, dict) and data.get("datePublished"):
                    published = data.get("datePublished")
                    break
            except Exception:
                continue
    except Exception:
        pass

    # fallback: common meta tags
    if not published:
        meta_keys = ["article:published_time", "og:published_time", "publication_date", "pubdate", "date"]
        for k in meta_keys:
            tag = soup.find("meta", attrs={"property": k}) or soup.find("meta", attrs={"name": k})
            if tag and tag.get("content"):
                published = tag.get("content")
                break

    texts = []
    for el in soup.find_all(["h1", "h2", "h3", "p"]):
        txt = el.get_text(separator=" ", strip=True)
        if txt:
            texts.append(txt)
    text = "\n\n".join(texts)
    if len(text) > 200_000:
        text = text[:200_000]

    pub_iso = None
    if published:
        try:
            dt = dateparser.parse(published)
            pub_iso = dt.isoformat()
        except Exception:
            pub_iso = published

    return {"title": title, "text": text, "published": pub_iso}


def _parse_tavily_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in data.get("results", []) if isinstance(data, dict) else []:
        url = item.get("url")
        if not url:
            continue
        out.append({
            "id": _short_id(),
            "url": url,
            "domain": _domain_from_url(url),
            "title": item.get("title") or "",
            "snippet": item.get("content") or "",
            "score": float(item.get("score", 0) or 0),
            "published": item.get("published") or None,
            "fetched": False,
            "full_text": None
        })
    return out


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY missing; set in env or backend.config.cfg")

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results
    }
    try:
        resp = httpx.post(TAVILY_SEARCH_URL, json=payload, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        return _parse_tavily_results(data)
    except Exception as e:
        logger.exception("Tavily search failed for query=%s : %s", query, e)
        return []


async def _fetch_page_async(client: httpx.AsyncClient, url: str) -> Optional[httpx.Response]:
    if not _is_safe_scheme(url):
        return None
    for attempt in range(FETCH_RETRIES + 1):
        try:
            resp = await client.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200 and resp.headers.get("content-type", "").lower().startswith(("text/html", "application/xhtml+xml")):
                return resp
            if resp.status_code in (403, 404):
                logger.debug("Fetch got %s for %s", resp.status_code, url)
                return resp
        except httpx.ForbiddenError:
            return None
        except Exception as e:
            logger.debug("fetch attempt %s failed for %s: %s", attempt, url, e)
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def _fetch_and_extract_all(urls: List[str],
                                 fetch_full: bool = False,
                                 user_id: Optional[str] = None,
                                 user_plan: str = "free") -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    timeout = httpx.Timeout(HTTP_TIMEOUT, read=HTTP_TIMEOUT)
    limits = httpx.Limits(max_connections=MAX_CONCURRENT_FETCHES, max_keepalive_connections=MAX_CONCURRENT_FETCHES)
    headers = {"User-Agent": USER_AGENT}

    sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers, http2=False, verify=True) as client:

        async def _fetch(url: str):
            # Early reject if scheme unsafe
            if not _is_safe_scheme(url):
                results[url] = {"title": "", "text": "", "published": None, "fetched": False, "rate_limited": False}
                return

            # Check cache first for full fetch requests
            if fetch_full:
                cached_content = get_cached_content(url)
                if cached_content:
                    logger.debug("Cache hit for URL: %s", url)
                    results[url] = {
                        "title": cached_content.get("title", ""),
                        "text": cached_content.get("text", ""),
                        "published": cached_content.get("published"),
                        "fetched": True,
                        "cached": True
                    }
                    return

            domain = _domain_from_url(url)

            # Distributed domain limiter
            allowed = allow_domain_request(domain, max_requests=PER_DOMAIN_MAX, window_seconds=PER_DOMAIN_WINDOW)
            if not allowed:
                results[url] = {"title": "", "text": "", "published": None, "fetched": False, "rate_limited": True}
                return

            # local cooldown
            ensure_local_cooldown(domain, PER_DOMAIN_MIN_INTERVAL)

            # If full fetch is requested, check user quota (consume 1)
            if fetch_full:
                quota_limit = PLAN_QUOTAS.get(user_plan, 0)
                ok_user = consume_user_quota(user_id or "", quota_limit)
                if not ok_user:
                    results[url] = {"title": "", "text": "", "published": None, "fetched": False, "user_quota_exhausted": True}
                    return

            try:
                async with sem:
                    resp = await _fetch_page_async(client, url)
                    if resp is None:
                        results[url] = {"title": "", "text": "", "published": None, "fetched": False, "rate_limited": False}
                        return
                    html = resp.text or ""
                    extracted = _extract_text_from_html(html, base_url=url)
                    extracted["fetched"] = True
                    extracted["cached"] = False
                    
                    # Cache the fetched content for future use
                    if fetch_full and extracted.get("text"):
                        set_cached_content(
                            url=url,
                            title=extracted.get("title", ""),
                            text=extracted.get("text", ""),
                            published=extracted.get("published")
                        )
                    
                    results[url] = extracted
            except Exception as e:
                logger.debug("Error fetching %s: %s", url, e)
                results[url] = {"title": "", "text": "", "published": None, "fetched": False, "rate_limited": False}

        tasks = []
        for u in urls:
            if fetch_full:
                tasks.append(asyncio.create_task(_fetch(u)))
            else:
                results[u] = {"title": "", "text": "", "published": None, "fetched": False}
        if tasks:
            await asyncio.gather(*tasks)
    return results


def scout(subqueries: List[str],
          max_urls_per_query: int = 5,
          fetch_full: bool = False,
          user_id: Optional[str] = None,
          user_plan: str = "free") -> List[Dict[str, Any]]:
    """
    Synchronous wrapper used by pipeline.

    - user_id: optional string for quota enforcement (required for fetch_full)
    - user_plan: "free"|"pro"|"premium" (controls monthly quota)
    """
    all_hits: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    for q in subqueries:
        try:
            hits = tavily_search(q, max_results=max_urls_per_query)
        except Exception as e:
            logger.exception("Tavily error for q=%s: %s", q, e)
            hits = []
        for h in hits:
            url = h.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            all_hits.append(h)

    if fetch_full and all_hits:
        urls = [h["url"] for h in all_hits]
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            fetched_map = loop.run_until_complete(_fetch_and_extract_all(urls, fetch_full=True, user_id=user_id, user_plan=user_plan))
        finally:
            try:
                loop.close()
            except Exception:
                pass
        for h in all_hits:
            fm = fetched_map.get(h["url"], {})
            h["title"] = h.get("title") or fm.get("title") or ""
            h["full_text"] = fm.get("text")
            h["published"] = h.get("published") or fm.get("published")
            h["fetched"] = fm.get("fetched", False)
            h["cached"] = fm.get("cached", False)
            # propagate limiter flags if present
            for k in ("rate_limited", "user_quota_exhausted"):
                if fm.get(k):
                    h[k] = True
    else:
        for h in all_hits:
            h.setdefault("full_text", None)
            h.setdefault("fetched", False)
            h.setdefault("cached", False)

    logger.info("Scout: returning %d sources (fetch_full=%s)", len(all_hits), bool(fetch_full))
    return all_hits


# Quick CLI for manual testing
if __name__ == "__main__":
    qlist = ["enterprise agent orchestration frameworks 2025 pricing", "langgraph vs autogen features"]
    out = scout(qlist, max_urls_per_query=3, fetch_full=False)
    print(json.dumps(out, indent=2)[:2000])
