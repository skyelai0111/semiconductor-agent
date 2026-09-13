import os
import streamlit as st
from dotenv import load_dotenv
from step_query import extract_query
from test_arxiv import search_papers
from test_llm import process_papers

load_dotenv()

st.set_page_config(page_title="半导体文献检索助手", layout="wide")
st.title("半导体科研文献检索助手")
st.caption("描述你的工作上下文，Agent 会检索文献并生成带推荐理由的证据卡片。")

user_input = st.text_area(
    "描述你的工作上下文",
    placeholder="例如：我想用机器学习预测射频电路的S参数，输入是版图的二维矩阵，目标是加速电磁仿真并支持后续优化。",
    height=120
)

if st.button("开始检索", type="primary"):
    if not user_input.strip():
        st.warning("请先输入你的工作描述。")
        st.stop()

    # 1. 抽取查询词
    with st.spinner("正在从你的描述中抽取检索词..."):
        try:
            query = extract_query(user_input)
        except Exception as e:
            st.error(f"查询词抽取失败：{e}")
            st.stop()

    st.info(f"抽取的检索表达式：`{query}`")

    # 2. 检索文献
    with st.spinner("正在检索 OpenAlex..."):
        try:
            papers = search_papers(query, max_results=50)
        except Exception as e:
            st.error(f"检索失败：{e}")
            st.stop()

    if not papers:
        st.warning("没有检索到论文。尝试换一种描述方式，或放宽关键词。")
        st.stop()

    st.success(f"检索到 {len(papers)} 篇论文，开始逐篇分析...")

    # 3. LLM 分析（两阶段）
    progress = st.progress(0, text="正在分析相关性...")
    try:
        results = process_papers(user_input, papers)
    except Exception as e:
        st.error(f"分析过程出错：{e}")
        st.stop()
    progress.progress(100, text="分析完成")

    if not results:
        st.warning("没有论文通过相关性初筛。可以尝试调整你的描述，或放宽检索条件。")
        st.stop()

    # 4. 排序并展示
    order = {"高": 0, "中": 1, "低": 2, "未知": 3}

    def sort_key(r):
        transfer = order.get(r.get("transferability", "未知"), 3)
        year = int(r["paper"].get("published", "0") or "0")
        year_score = 0 if year >= 2026 else (1 if year >= 2025 else 2)
        return (transfer, year_score)

    sorted_results = sorted(results, key=sort_key)

    st.subheader("推荐文献")
    st.caption("按可迁移性优先、年份次之排序，最多展示 5 篇。")

    for r in sorted_results[:5]:
        p = r["paper"]
        title = f"{p['title']} ({p['published']})"
        with st.expander(title, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**来源**：{p.get('venue', '未知')}")
                st.write(f"**引用数**：{p.get('citations', 0)}")
                st.write(f"**任务类型**：{r.get('task_type', '未知')}")
                st.write(f"**ML 方法**：{r.get('ml_method', '未知')}")
            with col2:
                st.write(f"**可迁移性**：{r.get('transferability', '未知')}")
                st.write(f"**信息充分**：{r.get('info_sufficient', '未知')}")
                st.write(f"**数据表示**：{r.get('data_representation', '未知')}")

            st.markdown(f"**推荐理由**：{r.get('reason', '无')}")
            if p.get("url"):
                st.markdown(f"[查看原文]({p['url']})")
