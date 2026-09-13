import os
import requests
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


def fetch_raw_works(query: str, max_results: int = 3) -> list[dict]:
    """直接请求 OpenAlex，返回原始 JSON 结果。"""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": max_results,
        "filter": "has_abstract:true",
        "sort": "relevance_score:desc",
        "mailto": "your_email@example.com"
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def verify(query: str, max_results: int = 3):
    works = fetch_raw_works(query, max_results)
    print(f"查询：{query}")
    print(f"返回论文数：{len(works)}\n")
    print("=" * 80)

    for i, work in enumerate(works, 1):
        title = work.get("title", "Untitled")
        year = work.get("publication_year", "?")
        inverted_index = work.get("abstract_inverted_index")

        has_index = inverted_index is not None and len(inverted_index) > 0
        abstract = reconstruct_abstract(inverted_index) if has_index else ""
        char_count = len(abstract)
        word_count = len(abstract.split()) if abstract else 0

        keywords = ["FinFET", "DIBL", "short channel", "TCAD", "drain", "gate"]
        found_keywords = [kw for kw in keywords if kw.lower() in abstract.lower()]

        print(f"\n[{i}] {title} ({year})")
        print(f"倒排索引存在：{has_index}")
        print(f"字符数：{char_count} | 词数：{word_count}")
        print(f"命中关键词：{found_keywords if found_keywords else '无'}")
        print(f"\n--- 完整摘要 ---\n")
        print(abstract if abstract else "（无摘要）")
        print("\n" + "-" * 80)


if __name__ == "__main__":
    verify("FinFET DIBL optimization", max_results=3)
