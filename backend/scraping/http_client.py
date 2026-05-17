from __future__ import annotations

import os
import random
import time
from typing import Sequence

import httpx

DEFAULT_USER_AGENTS: Sequence[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
)


def build_user_agent_rotation() -> list[str]:
    custom = os.environ.get("HTTP_USER_AGENT", "").strip()
    agents = list(DEFAULT_USER_AGENTS)
    if custom:
        agents.insert(0, custom)
    return agents


def jitter_sleep(min_s: float = 0.35, max_s: float = 1.2) -> None:
    time.sleep(random.uniform(min_s, max_s))


def fetch_get(
    url: str,
    *,
    timeout_s: float = 30.0,
) -> httpx.Response:
    """GET with jitter, rotating User-Agent, and sensible defaults."""
    jitter_sleep()
    ua = random.choice(build_user_agent_rotation())
    headers = {"User-Agent": ua, "Accept": "application/json, text/html;q=0.9,*/*;q=0.8"}
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        return client.get(url, headers=headers)
