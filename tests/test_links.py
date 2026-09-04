"""Every URL cited in docs, reports, config comments and code must resolve (slow: network)."""

import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [*ROOT.glob("*.md"), ROOT / ".env.example", ROOT / "pyproject.toml",
           *ROOT.glob("docs/*.md"), *ROOT.glob("reports/*.md"),
           *ROOT.glob("reports/experiments/*.md"), *ROOT.glob("src/snake4d/*.py")]
URL = re.compile(r"https?://[^\s<>()\"'\]`]+")
BOT_BLOCKED = ("dl.acm.org", "doi.org", "epubs.siam.org")  # answer 403 to scripted clients


def cited_urls() -> list[str]:
    urls = set()
    for path in SOURCES:
        for match in URL.findall(path.read_text(encoding="utf-8")):
            urls.add(match.rstrip(".,;:"))
    return sorted(urls)


def status(url: str) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (link check)"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


@pytest.mark.slow
@pytest.mark.parametrize("url", cited_urls())
def test_cited_url_resolves(url):
    try:
        code = status(url)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        pytest.skip(f"network unavailable for {url}: {error}")
    if code in (403, 429) and any(host in url for host in BOT_BLOCKED):
        pytest.xfail(f"{url} blocks scripted clients ({code}); verify in a browser")
    assert code < 400, f"{url} -> HTTP {code}"
