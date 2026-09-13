import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from test_arxiv import search_papers
from step_query import extract_query

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)


def quick_relevance_check(user_input: str, paper: dict) -> str:
    """轻量判断：这篇论文是否值得进入深度分析。返回 '是' / '否'。"""
    prompt = f"""用户需求：{user_input}

论文标题：{paper['title']}
论文摘要：{paper['abstract'][:500]}

这篇论文是否与用户需求相关？只回答"是"或"否"。"""
    response = llm.invoke(prompt).content.strip()
    return "是" if "是" in response else "否"


def analyze_paper(user_input: str, paper: dict) -> dict:
    """深度分析：从论文摘要中提取结构化字段，并生成推荐理由。"""
    prompt = f"""你是一个电路设计自动化（EDA）领域的文献分析助手。用户正在做以下工作：

用户上下文：{user_input}

候选论文信息：
- 标题：{paper['title']}
- 摘要：{paper['abstract']}
- 来源：{paper['venue']}
- 年份：{paper['published']}
- 引用数：{paper['citations']}

请从摘要中提取以下信息，以 JSON 格式返回。如果摘要信息不足以判断某个字段，填 "未知"。

{{
  "task_type": "核心任务类型（S参数预测 / 逆设计 / 代理模型 / 性能预测 / 其他）",
  "ml_method": "使用的机器学习方法（CNN / Transformer / 扩散模型 / 强化学习 / 其他）",
  "circuit_object": "电路对象（RFIC / 微波电路 / 传输线 / 无源器件 / 阻抗匹配 / 其他）",
  "data_representation": "数据表示方式（像素化/图像化 / 参数化 / 序列化 / 其他）",
  "optimization": "是否涉及优化算法（是 / 否 / 未知），如果是，简述优化目标",
  "key_result": "核心结论（一句话）",
  "transferability": "对用户问题的可迁移性（高 / 中 / 低）",
  "reason": "为什么这篇论文对用户当前问题有价值（一到两句话，引用论文中的具体方法和用户上下文中的具体目标）",
  "info_sufficient": "摘要信息是否足够支撑上述判断（是 / 否）"
}}

注意：
- 如果论文的输入表示是"电路版图的二维矩阵"或"像素化结构"，在 data_representation 中明确标注。
- 如果论文同时涉及 S 参数预测和后续优化，在 optimization 中说明优化目标。
- 如果摘要信息不足但论文发表于 2025 年之后，在 reason 中标注"新近发表，摘要数据可能不完整，建议查看原文"。

只返回 JSON，不要其他内容。"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {e}", "raw": content}


def process_papers(user_input: str, papers: list[dict]) -> list[dict]:
    """两阶段处理：先轻量筛选，再深度分析。"""
    results = []
    total = len(papers)
    passed = 0

    for i, paper in enumerate(papers, 1):
        title_short = paper["title"][:60]
        print(f"\n[{i}/{total}] {title_short} ({paper['published']})")

        # 无摘要的论文：跳过 LLM，生成占位结果
        if not paper["has_abstract"]:
            print("  → 无摘要，跳过")
            results.append({
                "paper": paper,
                "task_type": "未知",
                "ml_method": "未知",
                "circuit_object": "未知",
                "data_representation": "未知",
                "optimization": "未知",
                "key_result": "未知",
                "transferability": "未知",
                "reason": "新近发表，OpenAlex 尚未收录摘要，建议直接查看原文。",
                "info_sufficient": "否"
            })
            continue

        # 阶段一：轻量相关性判断
        quick = quick_relevance_check(user_input, paper)
        if quick == "否":
            print("  → 快速筛选：不相关，跳过")
            continue

        passed += 1
        print(f"  → 通过初筛，进行深度分析...")

        # 阶段二：深度分析
        result = analyze_paper(user_input, paper)
        if "error" in result:
            print(f"  → 分析失败：{result['error']}")
            continue

        result["paper"] = paper
        results.append(result)

        print(f"     任务类型：{result.get('task_type')}")
        print(f"     可迁移性：{result.get('transferability')}")
        print(f"     推荐理由：{result.get('reason')[:80]}...")

    print(f"\n初筛通过：{passed}/{total} 篇")
    return results


def sort_and_display(results: list[dict], top_n: int = 5):
    """按可迁移性 + 年份排序，展示前 N 篇。"""
    order = {"高": 0, "中": 1, "低": 2, "未知": 3}

    def sort_key(r):
        transfer_score = order.get(r.get("transferability", "未知"), 3)
        year = int(r["paper"].get("published", "0") or "0")
        year_score = 0 if year >= 2026 else (1 if year >= 2025 else 2)
        return (transfer_score, year_score)

    sorted_results = sorted(results, key=sort_key)

    print("\n" + "=" * 80)
    print(f"\n【按可迁移性 + 年份排序后的前 {top_n} 篇】\n")

    for i, r in enumerate(sorted_results[:top_n], 1):
        p = r["paper"]
        print(f"[{i}] {p['title']} ({p['published']})")
        print(f"    来源：{p['venue']} | 引用：{p['citations']} | 有摘要：{p['has_abstract']}")
        print(f"    任务类型：{r.get('task_type')} | ML方法：{r.get('ml_method')}")
        print(f"    电路对象：{r.get('circuit_object')} | 数据表示：{r.get('data_representation')}")
        print(f"    可迁移性：{r.get('transferability')} | 信息充分：{r.get('info_sufficient')}")
        print(f"    推荐理由：{r.get('reason')}")
        print(f"    {p['url']}\n")


if __name__ == "__main__":
    user_input = "我想用机器学习方法预测射频电路的S参数，输入是电路版图的二维矩阵表示，目标是加速电磁仿真"
    query = extract_query(user_input)
    print(f"抽取的检索表达式：{query}\n")
    papers = search_papers(query, max_results=50)

    print(f"检索到 {len(papers)} 篇论文，开始两阶段分析...\n")
    print("=" * 80)

    results = process_papers(user_input, papers)
    sort_and_display(results, top_n=5)
