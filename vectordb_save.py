import logging
from openai import OpenAI
import chromadb
import uuid
import numpy as np
import argparse
from tools import pdfSplitTest_Ch
from tools import pdfSplitTest_En

# 设置日志模板
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def argparser():
    parser = argparse.ArgumentParser(description="PDF向量化存储工具")

    # 模型设置相关
    parser.add_argument("--api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="OpenAI API 地址")
    parser.add_argument("--api_key", type=str, default="",
                        help="OpenAI API Key")
    parser.add_argument("--embed_model", type=str, default="text-embedding-v2", help="嵌入模型名称")

    # 语言与文件相关
    parser.add_argument("--language", type=str, default="Chinese", choices=["Chinese", "English"], help="文本语言")
    parser.add_argument("--input_pdf", type=str, default="input_pdf/健康档案.pdf", help="PDF文件路径")
    parser.add_argument("--page_numbers", type=str, default=None,
                        help="指定页码(逗号分隔，如'2,3'，默认全部页码)")

    # 向量数据库相关
    parser.add_argument("--chroma_dir", type=str, default="./chromaDB", help="向量数据库存储路径")
    parser.add_argument("--collection", type=str, default="demo001", help="向量集合名称")

    # 查询参数
    parser.add_argument("--query_text", type=str, default="张三九的血糖数值", help="测试查询文本")
    parser.add_argument("--query_num", type=int, default=3, help="查询返回结果数量")

    return parser.parse_args()


def get_embeddings(texts, api_base, api_key, embed_model):
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        data = client.embeddings.create(input=texts, model=embed_model).data
        return [x.embedding for x in data]
    except Exception as e:
        logger.error(f"生成向量时出错: {e}")
        return []


def generate_vectors(data, api_base, api_key, embed_model, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        response = get_embeddings(batch, api_base, api_key, embed_model)
        results.extend(response)
    return results


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn, persistence_path):
        chroma_client = chromadb.PersistentClient(path=persistence_path)
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            documents=documents,
            ids=[str(uuid.uuid4()) for _ in range(len(documents))]
        )

    def search(self, query, top_n):
        try:
            results = self.collection.query(
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            return results
        except Exception as e:
            logger.error(f"检索向量数据库时出错: {e}")
            return []


def vectorStoreSave(args):
    # 解析页码参数
    page_numbers = None
    if args.page_numbers:
        page_numbers = [int(num) for num in args.page_numbers.split(",")]

    # 创建向量生成函数（闭包）
    def embedding_fn(texts):
        return generate_vectors(
            texts,
            args.api_base,
            args.api_key,
            args.embed_model
        )

    # 处理PDF文件
    if args.language == 'Chinese':
        paragraphs = pdfSplitTest_Ch.getParagraphs(
            filename=args.input_pdf,
            page_numbers=page_numbers,
            min_line_length=1
        )
    elif args.language == 'English':
        paragraphs = pdfSplitTest_En.getParagraphs(
            filename=args.input_pdf,
            page_numbers=page_numbers,
            min_line_length=1
        )
    else:
        logger.error(f"不支持的语言: {args.language}")
        return

    # 存储到向量数据库
    vector_db = MyVectorDBConnector(
        collection_name=args.collection,
        embedding_fn=embedding_fn,
        persistence_path=args.chroma_dir
    )
    vector_db.add_documents(paragraphs)
    logger.info(f"成功存储 {len(paragraphs)} 个段落到向量数据库")

    # 执行查询测试
    logger.info(f"召回测试query: {args.query_text}")
    if args.query_text:
        search_results = vector_db.search(args.query_text, args.query_num)
        logger.info("向量数据库检索结果:")
        for i, doc in enumerate(search_results['documents'][0]):
            logger.info(f"结果 {i + 1}:\n{doc}\n{'-' * 50}")


if __name__ == "__main__":
    args = argparser()
    vectorStoreSave(args)