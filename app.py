import streamlit as st
from step_query import extract_query
from test_arxiv import search_papers
from test_llm import process_papers, sort_and_display

st.title("半导体科研文献检索助手")

user_input = st.text_area(
    "描述你的工作上下文",
    placeholder="例如：我想用机器学习预测射频电路的S参数，输入是版图的二维矩阵..."
)

if st.button("开始检索"):
    if not user_input.strip():
        st.warning("请输入你的工作描述")
    else:
        with st.spinner("正在抽取查询词..."):
            query = extract_query(user_input)
            st.caption(f"检索表达式：{query}")
        
        with st.spinner("正在检索文献..."):
            papers = search_papers(query, max_results=50)
            st.info(f"检索到 {len(papers)} 篇论文")
        
        with st.spinner("正在分析相关性（约需1-2分钟）..."):
            results = process_papers(user_input, papers)
        
        # 按可迁移性排序，展示前5篇
        order = {"高": 0, "中": 1, "低": 2, "未知": 3}
        def sort_key(r):
            transfer = order.get(r.get("transferability", "未知"), 3)
            year = int(r["paper"].get("published", "0") or "0")
            year_score = 0 if year >= 2026 else (1 if year >= 2025 else 2)
            return (transfer, year_score)
        
        sorted_results = sorted(results, key=sort_key)
        
        for r in sorted_results[:5]:
            p = r["paper"]
            with st.expander(f"**{p['title']}** ({p['published']})"):
                st.write(f"**来源**：{p['venue']} | **引用**：{p['citations']}")
                st.write(f"**可迁移性**：{r.get('transferability')} | **信息充分**：{r.get('info_sufficient')}")
                st.write(f"**推荐理由**：{r.get('reason')}")
                st.write(f"[查看原文]({p['url']})")
