"""DuckDuckGo 文本搜索，供多个助手页面复用。"""
from __future__ import annotations

from typing import Any, List, Tuple

try:
    from duckduckgo_search import DDGS

    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False
    DDGS = None


def search_web(query: str, max_results: int = 5) -> Tuple[str, List[dict[str, Any]]]:
    """
    执行联网搜索。

    Returns:
        (格式化的搜索结果字符串, 结果字典列表)。未安装依赖或失败时返回 ("", [])。
    """
    if not HAS_DUCKDUCKGO:
        return "", []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "", []

        search_results_text: List[str] = []
        search_results_list: List[dict[str, Any]] = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            body = result.get("body", "无内容")
            href = result.get("href", "")
            search_results_text.append(
                f"【搜索结果 {i}】{title}\n"
                f"链接: {href}\n"
                f"内容: {body}\n"
            )
            search_results_list.append({"title": title, "href": href, "body": body})

        return "\n".join(search_results_text), search_results_list
    except Exception:
        return "", []

