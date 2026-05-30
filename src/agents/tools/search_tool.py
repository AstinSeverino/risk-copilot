"""DuckDuckGo web search wrapper for the context researcher agent."""
from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return [{"title": "No results found", "body": f"No web results for: {query}", "href": ""}]
        return [
            {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
            for r in results
        ]
    except Exception as e:
        return [{"title": "Search unavailable", "body": str(e), "href": ""}]
