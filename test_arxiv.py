import os
import requests
import time
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")


def reconstruct_abstract(inverted_index: dict) -> str:
    """从 OpenAlex 的倒排索引中重建摘要文本。"""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def _parse_work(work: dict) -> dict:
    """把 OpenAlex 的原始 work 对象解析成统一格式。"""
    authors = [
        auth.get("author", {}).get("display_name", "")
        for auth in work.get("authorships", [])
    ]
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name", "")

    abstract = reconstruct_abstract(work.get("abstract_inverted_index", {}))
    has_abstract = len(abstract) > 0

    return {
        "id": work.get("id", ""),
        "title": work.get("title", "Untitled"),
        "abstract": abstract[:2000] if has_abstract else "",
        "has_abstract": has_abstract,
        "url": work.get("doi") or work.get("id", ""),
        "published": str(work.get("publication_year", "")),
        "venue": venue,
        "authors": authors[:3],
        "citations": work.get("cited_by_count", 0)
    }


def _request_openalex(params: dict, max_attempts: int = 3) -> list[dict]:
    """发送 OpenAlex 请求，带重试机制，返回解析后的结果列表。"""
    url = "https://api.openalex.org/works"
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                print(f"限流，{delay}秒后重试 ({attempt}/{max_attempts})")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            return [_parse_work(w) for w in data.get("results", [])]
        except requests.RequestException as e:
            if attempt < max_attempts:
                print(f"请求失败: {e}，{delay}秒后重试")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return []


def search_papers(query: str, max_results: int = 50) -> list[dict]:
    """
    组合检索：
    - 策略一：35 篇相关性优先（要求有摘要，2024-01-01 之后）
    - 策略二：15 篇时效性补充（2026-01-01 之后，不限摘要）
    合并去重，返回最多 50 篇。
    """
    all_results = []
    seen_ids = set()

    # 策略一：相关性优先，有摘要
    params_coverage = {
        "search": query,
        "per_page": 35,
        "filter": "has_abstract:true,from_publication_date:2024-01-01",
        "sort": "relevance_score:desc",
        "mailto": "your_email@example.com"
    }
    coverage_results = _request_openalex(params_coverage)
    for work in coverage_results:
        if work["id"] not in seen_ids:
            seen_ids.add(work["id"])
            all_results.append(work)

    # 策略二：时效性补充，2026 年之后，不限摘要
    params_recency = {
        "search": query,
        "per_page": 15,
        "filter": "from_publication_date:2026-01-01",
        "sort": "publication_date:desc",
        "mailto": "your_email@example.com"
    }
    recency_results = _request_openalex(params_recency)
    for work in recency_results:
        if work["id"] not in seen_ids:
            seen_ids.add(work["id"])
            all_results.append(work)

    return all_results[:max_results]


if __name__ == "__main__":
    test_query = '("S-parameter" OR "scattering parameter") AND ("neural network" OR "machine learning" OR "deep learning") AND (prediction OR modeling OR surrogate)'
    papers = search_papers(test_query, max_results=50)

    print(f"共检索到 {len(papers)} 篇论文\n")

    years = Counter(p["published"] for p in papers)
    print("年份分布：", dict(sorted(years.items(), reverse=True)))
    has_abs = sum(1 for p in papers if p["has_abstract"])
    print(f"有摘要的论文：{has_abs} 篇，无摘要：{len(papers) - has_abs} 篇\n")

    # 显示前 20 篇
    for i, p in enumerate(papers[:20], 1):
        abs_mark = "✓" if p["has_abstract"] else "✗"
        print(f"[{i}] {p['title']} ({p['published']}) {abs_mark}")
        print(f"    来源: {p['venue']} | 引用: {p['citations']} | 摘要: {len(p['abstract'])} 字符")
        print(f"    {p['url']}\n")

    if len(papers) > 20:
        print(f"（仅显示前 20 篇，共 {len(papers)} 篇）")
