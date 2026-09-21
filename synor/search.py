"""
Hybrid Web Search & Real-Time Knowledge Grounding Engine for Synor AI.
Combines DuckDuckGo Instant Answer API, Wikipedia Knowledge Graph, and DuckDuckGo Web Search.
Zero API keys required.
"""

import html
import re
from typing import List, Optional
import requests


class WebSearchEngine:
    """
    High-reliability, zero-API-key web search engine.
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "SynorAI/1.0 (Educational Neural Assistant; macOS)"
        }

    def _search_ddg_instant(self, query: str) -> Optional[str]:
        """Try DuckDuckGo Instant Answer API."""
        try:
            r = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
                headers=self.headers,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                data = r.json()
                abstract = data.get("AbstractText", "").strip()
                if abstract and len(abstract) > 25:
                    heading = data.get("Heading", query)
                    return f"{heading}: {abstract}"
                answer = data.get("Answer", "").strip()
                if answer:
                    return answer
        except Exception:
            pass
        return None

    def _search_wikipedia(self, query: str) -> Optional[str]:
        """Search Wikipedia REST API for factual and encyclopedic grounding."""
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": 2,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                data = r.json()
                items = data.get("query", {}).get("search", [])
                if items:
                    results = []
                    for item in items[:2]:
                        title = item.get("title", "")
                        raw_snippet = item.get("snippet", "")
                        clean_snippet = re.sub(r"<[^<]+?>", "", raw_snippet)
                        unescaped = html.unescape(clean_snippet).strip()
                        if len(unescaped) > 20:
                            results.append(f"{title}: {unescaped}")
                    if results:
                        return "\n\n".join(results)
        except Exception:
            pass
        return None

    def _search_ddg_lite(self, query: str, max_results: int = 2) -> Optional[str]:
        """Fallback to DuckDuckGo Lite search."""
        try:
            r = requests.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=self.timeout,
            )
            if r.status_code == 200:
                matches = re.findall(
                    r"<td[^>]*class=[\'\"]?result-snippet[\'\"]?[^>]*>(.*?)</td>",
                    r.text,
                    re.DOTALL | re.IGNORECASE,
                )
                cleaned: List[str] = []
                for m in matches:
                    no_tags = re.sub(r"<[^<]+?>", "", m)
                    unescaped = html.unescape(no_tags).strip()
                    normalized = re.sub(r"\s+", " ", unescaped)
                    if len(normalized) > 25:
                        cleaned.append(normalized)

                query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
                relevant = [s for s in cleaned if any(w in s.lower() for w in query_words)]
                if relevant:
                    return "\n\n".join(relevant[:max_results])
        except Exception:
            pass
        return None

    def search(self, query: str) -> Optional[str]:
        """
        Execute unified web search across DuckDuckGo and Wikipedia.
        Returns a clean summary or None if completely unknown/unreachable.
        """
        clean_query = query.strip()
        if not clean_query:
            return None

        # 1. DuckDuckGo Instant Answer
        res = self._search_ddg_instant(clean_query)
        if res:
            return res

        # 2. Wikipedia Knowledge API
        res = self._search_wikipedia(clean_query)
        if res:
            return res

        # 3. DuckDuckGo Lite Search
        res = self._search_ddg_lite(clean_query)
        if res:
            return res

        return None


# Global singleton
search_engine = WebSearchEngine()


def should_search_web(prompt: str) -> bool:
    """
    Determine if a user prompt is asking for factual, real-world, or external knowledge.
    """
    p = prompt.lower().strip()
    if p.startswith(("/search", "search ", "google ", "find ", "who is ", "who was ", "what is ", "when was ", "where is ", "how to ")):
        return True
    triggers = [
        "ceo of",
        "founder of",
        "capital of",
        "president of",
        "prime minister of",
        "weather",
        "latest news",
        "price of",
        "meaning of",
        "who created",
        "population of",
        "history of",
    ]
    return any(t in p for t in triggers)
