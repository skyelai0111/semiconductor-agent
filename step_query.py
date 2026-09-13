import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)


def extract_query(user_input: str) -> str:
    """把用户的自然语言描述翻译成学术检索表达式。"""
    prompt = f"""你是一个学术文献检索专家。用户描述了他的工作需求，请从中抽取适合用于 OpenAlex 检索的英文查询表达式。

规则：
- 核心任务用最精确的术语，比如 "S-parameter prediction" 而不是 "predict S-parameter"
- 方法关键词用 OR 扩展同义词，比如 ("neural network" OR "deep learning" OR "surrogate model")
- 应用对象也用 OR 扩展，比如 ("RFIC" OR "microwave circuit" OR "integrated circuit")
- 不同维度之间用 AND 连接
- 不要加入用户没有提到的术语
- 只返回查询表达式，不要解释
- 整个表达式最多使用 2 个 AND
- 核心任务维度：用 OR 扩展 2-3 个同义词
- 方法维度：用 OR 扩展 3-5 个同义词
- 不要在查询中加入应用对象（如 RFIC、microwave circuit），这些留给后续分析阶段判断
- 不要使用短语匹配（如 "S-parameter prediction"），用单词或双词组合（如 "S-parameter"）
- 只返回查询表达式，不要解释

示例：
用户描述：我想用机器学习预测微波电路的S参数
返回：("S-parameter prediction" OR "S-parameter modeling") AND ("machine learning" OR "neural network" OR "deep learning") AND ("microwave circuit" OR "RF circuit")

用户描述：{user_input}
返回："""

    response = llm.invoke(prompt).content.strip()
    # 去掉可能的引号或 markdown 标记
    response = response.strip('"').strip("'").strip("`")
    return response


if __name__ == "__main__":
    test_inputs = [
        "我想用机器学习方法预测射频电路的S参数，输入是电路版图的二维矩阵表示，目标是加速电磁仿真并支持后续的优化算法",
        "想找用深度学习做微波滤波器S参数建模的论文",
        "AI辅助电路分析，涉及二维矩阵预测S参数的模型和优化算法"
    ]
    for inp in test_inputs:
        print(f"输入：{inp}")
        print(f"输出：{extract_query(inp)}\n")
