from __future__ import annotations

import asyncio
import contextlib
import io
import ipaddress
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_FETCH_MAX_CHARS = 30_000
DEFAULT_AGENTIC_MAX_TURNS = 15
MAX_FETCH_BYTES = 2_000_000


def validate_web_search_configuration(
    max_results: int,
    timeout_seconds: int,
    fetch_max_chars: int = DEFAULT_FETCH_MAX_CHARS,
    agentic_max_turns: int = DEFAULT_AGENTIC_MAX_TURNS,
) -> None:
    if not 1 <= max_results <= 20:
        raise ValueError("WEB_SEARCH_MAX_RESULTS must be between 1 and 20")
    if not 1 <= timeout_seconds <= 120:
        raise ValueError("WEB_SEARCH_TIMEOUT must be between 1 and 120 seconds")
    if not 2_000 <= fetch_max_chars <= 100_000:
        raise ValueError("WEB_FETCH_MAX_CHARS must be between 2000 and 100000")
    if not 2 <= agentic_max_turns <= 50:
        raise ValueError("WEB_SEARCH_MAX_TURNS must be between 2 and 50")


def build_web_search_arguments(
    enabled: bool,
    max_results: int,
    timeout_seconds: int,
    fetch_max_chars: int = DEFAULT_FETCH_MAX_CHARS,
    agentic_max_turns: int = DEFAULT_AGENTIC_MAX_TURNS,
    *,
    script_path: Path | None = None,
    python_executable: str | None = None,
) -> list[str]:
    if not enabled:
        return []

    validate_web_search_configuration(
        max_results,
        timeout_seconds,
        fetch_max_chars,
        agentic_max_turns,
    )
    server_script = (script_path or Path(__file__)).resolve()
    mcp_configuration = {
        "mcpServers": {
            "web-research": {
                "command": python_executable or sys.executable,
                "args": [str(server_script)],
                "env": {
                    "WEB_SEARCH_MAX_RESULTS": str(max_results),
                    "WEB_SEARCH_TIMEOUT": str(timeout_seconds),
                    "WEB_FETCH_MAX_CHARS": str(fetch_max_chars),
                },
            },
        }
    }
    return [
        "--jinja",
        "--ui-mcp-proxy",
        "--mcp-servers-json",
        json.dumps(mcp_configuration, separators=(",", ":")),
        "--ui-config",
        json.dumps(
            {
                "agenticMaxTurns": agentic_max_turns,
                "alwaysShowAgenticTurns": True,
                "showToolCallInProgress": True,
            },
            separators=(",", ":"),
        ),
    ]


def _normalize_results(raw_results: Any, max_results: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not isinstance(raw_results, list):
        return results
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", item.get("href", ""))).strip()
        title = str(item.get("title", "")).strip()
        if not url or not title:
            continue
        engines = item.get("engines", item.get("engine", []))
        if isinstance(engines, str):
            engines = [engines]
        results.append(
            {
                "title": title[:500],
                "url": url,
                "content": str(item.get("content", item.get("body", ""))).strip()[:2000],
                "engines": engines if isinstance(engines, list) else [],
                "score": item.get("score"),
            }
        )
        if len(results) >= max_results:
            break
    return results


def _search_duckduckgo(
    query: str, max_results: int, timeout_seconds: int
) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError as error:
        raise RuntimeError(
            "Web search requires 'ddgs'. Run: pip install -r requirements.txt"
        ) from error
    # Some backends emit diagnostics directly to stderr. Keep the MCP protocol
    # and the manager's recent-output view clean.
    with contextlib.redirect_stderr(io.StringIO()):
        raw_results = DDGS(timeout=timeout_seconds).text(
            query,
            safesearch="moderate",
            max_results=max_results,
            backend="auto",
        )
    return _normalize_results(raw_results, max_results)


def search_web(
    query: str,
    *,
    max_results: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty")
    if len(query) > 500:
        raise ValueError("Search query cannot exceed 500 characters")

    validate_web_search_configuration(max_results, timeout_seconds)
    try:
        results = _search_duckduckgo(query, max_results, timeout_seconds)
    except Exception as error:
        raise RuntimeError(f"Web search failed: {error}") from error
    if not results:
        raise RuntimeError("Web search returned no results")
    return {"query": query, "provider": "DuckDuckGo", "results": results}


def _validate_public_url(url: str) -> str:
    url = url.strip()
    if len(url) > 2_048:
        raise ValueError("URL cannot exceed 2048 characters")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("fetch_url accepts only complete http:// or https:// URLs")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise ValueError("URL contains an invalid port") from error
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError(f"Could not resolve URL hostname: {parsed.hostname}") from error
    if not addresses:
        raise ValueError(f"Could not resolve URL hostname: {parsed.hostname}")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("fetch_url cannot access local or private network addresses")
    return url


class _PublicOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        safe_url = _validate_public_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, safe_url
        )


def _clean_html(
    data: bytes,
    final_url: str,
    max_chars: int,
) -> tuple[str, str, list[dict[str, str]]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise RuntimeError(
            "URL fetching requires beautifulsoup4. Run: pip install -r requirements.txt"
        ) from error

    soup = BeautifulSoup(data, "html.parser")
    title = soup.title.get_text(" ", strip=True)[:500] if soup.title else ""
    for unwanted in soup(
        ["script", "style", "noscript", "svg", "canvas", "form", "nav", "footer"]
    ):
        unwanted.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    lines: list[str] = []
    previous = ""
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if line and line != previous:
            lines.append(line)
            previous = line
    cleaned_text = "\n".join(lines)

    links: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for anchor in root.find_all("a", href=True):
        href = urljoin(final_url, str(anchor["href"]).strip())
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"}:
            continue
        href = parsed._replace(fragment="").geturl()
        if href in seen_urls:
            continue
        label = " ".join(anchor.get_text(" ", strip=True).split())[:300]
        if not label:
            continue
        seen_urls.add(href)
        links.append({"text": label, "url": href})
        if len(links) >= 100:
            break
    return title, cleaned_text[:max_chars], links


def fetch_url(
    url: str,
    *,
    timeout_seconds: int,
    max_chars: int,
) -> dict[str, Any]:
    safe_url = _validate_public_url(url)
    opener = build_opener(_PublicOnlyRedirectHandler())
    request = Request(
        safe_url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "identity",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) llama.cpp-research-agent/1.0",
        },
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            final_url = _validate_public_url(response.geturl())
            content_type = response.headers.get_content_type().lower()
            data = response.read(MAX_FETCH_BYTES + 1)
    except HTTPError as error:
        raise RuntimeError(f"URL returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"Could not fetch URL: {error.reason}") from error
    except OSError as error:
        raise RuntimeError(f"Could not fetch URL: {error}") from error

    if len(data) > MAX_FETCH_BYTES:
        raise RuntimeError(f"URL response exceeds the {MAX_FETCH_BYTES}-byte safety limit")

    if content_type in {"text/html", "application/xhtml+xml"}:
        title, text, links = _clean_html(data, final_url, max_chars)
    elif content_type.startswith("text/") or content_type in {
        "application/json",
        "application/ld+json",
        "application/xml",
        "application/rss+xml",
        "application/atom+xml",
    }:
        title = ""
        text = data.decode("utf-8", "replace")[:max_chars]
        links = []
    else:
        raise RuntimeError(f"Unsupported URL content type: {content_type}")

    return {
        "url": final_url,
        "title": title,
        "content_type": content_type,
        "text": text,
        "truncated": len(text) >= max_chars,
        "links": links,
    }


def run_mcp_server() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError(
            "Web search requires the 'mcp' package. Run: pip install -r requirements.txt"
        ) from error

    max_results = int(
        os.environ.get("WEB_SEARCH_MAX_RESULTS", str(DEFAULT_MAX_RESULTS))
    )
    timeout_seconds = int(
        os.environ.get("WEB_SEARCH_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    )
    fetch_max_chars = int(
        os.environ.get("WEB_FETCH_MAX_CHARS", str(DEFAULT_FETCH_MAX_CHARS))
    )
    validate_web_search_configuration(
        max_results,
        timeout_seconds,
        fetch_max_chars,
    )

    server = FastMCP(
        "Web Research",
        instructions=(
            "For questions needing internet research, first call web_search to discover "
            "sources, then call fetch_url on the most relevant results. Continue calling "
            "web_search and fetch_url in successive turns until there is enough evidence. "
            "Do not answer from snippets alone when the underlying pages can be fetched. "
            "Cross-check important claims with additional sources and include source URLs "
            "in the final answer."
        ),
    )

    @server.tool()
    async def web_search(query: str) -> dict[str, Any]:
        """Discover web sources. After searching, use fetch_url on relevant result URLs before answering. This tool may be called repeatedly with refined queries."""
        return await asyncio.to_thread(
            search_web,
            query,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )

    @server.tool(name="fetch_url")
    async def fetch_url_tool(url: str) -> dict[str, Any]:
        """Open and extract readable text and links from a public web URL. Call this after web_search, and call it repeatedly for additional sources until the answer is adequately supported."""
        return await asyncio.to_thread(
            fetch_url,
            url,
            timeout_seconds=timeout_seconds,
            max_chars=fetch_max_chars,
        )

    server.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
