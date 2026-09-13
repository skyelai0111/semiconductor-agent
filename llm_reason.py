import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from test_arxiv import search_papers

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)


def analyze_paper(user_input: str, paper: dict) -> dict:
    """让 LLM 从论文摘要中提取结构化字段，并生成推荐理由。"""
    prompt = f"""你是一个半导体领域的文献分析助手。用户正在做以下工作：

用户上下文：{user_input}

候选论文信息：
- 标题：{paper['title']}
- 摘要：{paper['abstract']}
- 来源：{paper['venue']}
- 年份：{paper['published']}
- 引用数：{paper['citations']}

请从摘要中提取以下信息，以 JSON 格式返回。如果摘要信息不足以判断某个字段，填 "未知"。

{{
  "device_type": "器件类型（FinFET / GAA / Planar / 其他）",
  "method": "研究方法（TCAD仿真 / 实验测量 / 理论分析 / 综述 / 其他）",
  "target_issue": "针对的问题（DIBL / 短沟道效应 / 功耗 / 迁移率 / 其他）",
  "approach": "技术路径（结构参数调整 / 新材料 / 新工艺 / 电路级优化 / 其他）",
  "key_result": "核心结论（一句话）",
  "transferability": "对用户问题的可迁移性（高 / 中 / 低）",
  "reason": "为什么这篇论文对用户当前问题有价值（一到两句话，引用论文中的具体术语和用户上下文中的具体参数）",
  "info_sufficient": "摘要信息是否足够支撑上述判断（是 / 否）"
}}

只返回 JSON，不要其他内容。"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # 清理可能出现的 markdown 代码块标记
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {e}", "raw": content}


if __name__ == "__main__":
    user_input = "我在仿真28nm FinFET，Lg=20nm时DIBL超过100mV/V，想找结构优化方法"

    papers = search_papers("FinFET DIBL optimization", max_results=3)
    print(f"检索到 {len(papers)} 篇论文\n")
    print("=" * 80)

    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']} ({paper['published']})")
        print(f"来源：{paper['venue']} | 引用：{paper['citations']}")

        result = analyze_paper(user_input, paper)

        if "error" in result:
            print(f"分析失败：{result['error']}")
            print(f"原始输出：{result.get('raw', '')[:300]}")
            continue

        print(f"\n  器件类型：{result.get('device_type')}")
        print(f"  研究方法：{result.get('method')}")
        print(f"  针对问题：{result.get('target_issue')}")
        print(f"  技术路径：{result.get('approach')}")
        print(f"  核心结论：{result.get('key_result')}")
        print(f"  可迁移性：{result.get('transferability')}")
        print(f"  信息充分：{result.get('info_sufficient')}")
        print(f"  推荐理由：{result.get('reason')}")
        print("-" * 80)
