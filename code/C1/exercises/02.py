import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../../data/C1/markdown/easy-rl-chapter1.md" # 修改了路径

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# 默认：
# text_splitter = RecursiveCharacterTextSplitter()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)

# ['根据提供的上下文，文中举了以下例子：', '1.  探索与利用的例子：', '选择餐馆：利用是去熟悉的餐馆（知道菜好吃）；探索是搜索新餐馆尝试（可能浪费钱）。', '做广告：利用是采取最优广告策略；探索是换一种策略看效果。', '挖油：利用是在已知地方挖油（确保挖到）；探索是在新地方挖油（可能失败，也可能发现大油田）。', '玩游戏（以《街头霸王》为例）：利用是采取固定策略（如蹲在角落出脚）；探索是尝试新招式（如放“大招”）。', '2.  强化学习与监督学习的对比例子：', '图片分类：监督学习输入汽车、飞机、椅子等被标注的图片。', '3.  强化学习实验与应用的例子：', 'OpenAI 机械臂翻魔方：在虚拟环境训练后应用到真实机械臂。', '穿衣服的智能体：实现穿衣功能，抵抗扰动。', '4.  奖励的例子：', '象棋选手：赢棋得正奖励，输棋得负奖励。', '股票管理：奖励由股票收益与损失决定。', '玩雅达利游戏：奖励是游戏分数的增减。', '5.  序列决策与环境交互的例子：', '雅达利游戏 Pong：控制木板接球。', '雅达利游戏 Breakout：控制木板打砖块。', '6.  状态和观测表示的例子：', 'RGB 像素值的矩阵：表示视觉观测。', '机器人关节的角度和速度：表示机器人状态。']

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 6000, chunk_overlap = 200)
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)

# 修改Langchain代码中RecursiveCharacterTextSplitter()的参数chunk_size和chunk_overlap，观察输出结果有什么变化

# 实际上会出现的情况：

# 