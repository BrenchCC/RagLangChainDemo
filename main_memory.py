import os
import re
import json
import asyncio
import argparse
import uuid
import time
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
# 部署REST API相关
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
# 向量数据库chroma相关
from langchain_chroma import Chroma
# openai的向量模型
from langchain_openai import OpenAIEmbeddings
# RAG相关
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory # 处理和存储对话历史
from torchvision.transforms.v2.functional import ten_crop_image

# 设置langsmith变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL = None
EMBEDDINGS = None
VECTORSTORE = None
SYSTEM_PROMPT = None
CHAIN = None
PORT = 8002

PROMPT_TEMPLATE_TXT = "prompts/prompt.txt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default = "https://dashscope.aliyuncs.com/compatible-mode/v1", help = "OpenAI API URL")
    parser.add_argument("--api_key", default = "", help = "OpenAI API Key")
    parser.add_argument("--chat_model", default = "qwen-plus", help = "OpenAI Chat Model")
    parser.add_argument("--embedding_model", default = "text-embedding-v2", help = "OpenAI Chat Model ID")
    parser.add_argument("--chromadb_dir", default = "chromaDB", help = "Chromadb Directory")
    parser.add_argument("--chromadb_collection", default = "demov1", help = "Chroma Collection Name")

    return parser.parse_args()

# 定义Message类
class Message(BaseModel):
    role: str
    content: str

# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

# 定义ChatCompletionResponseChoice类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory = lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None

# 获取对话历史
def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}, sqlite:///memory/memory.db")

# 获取prompt在chain中传递的prompt最终的内容
def get_prompt(prompt):
    logger.info(f"最后给到LLM的prompt的内容: {prompt}")
    return prompt

def format_search_result(search_result):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', search_result)

    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []

    for paragraph in paragraphs:
        if "```" in paragraph:
            # 根据段落按照```分成多个部分，代码💨和普通文本交替出现
            parts = paragraph.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    # 代码块，用换行符和```包围，并去除多余空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            paragraph = " ".join(parts)
        #
        # else:
        #     # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
        #     paragraph = paragraph.replace('.', '.\n')
        formatted_paragraphs.append(paragraph.strip())

    return "\n\n".join(formatted_paragraphs)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, EMBEDDINGS, VECTORSTORE, SYSTEM_PROMPT, CHAIN, PORT

    parser = parse_args()
    base_url = parser.base_url
    api_key = parser.api_key
    chat_model = parser.chat_model
    embedding_model = parser.embedding_model
    chromadb_dir = parser.chromadb_dir
    chromadb_collection = parser.chromadb_collection

    try:
        logger.info("Initializing LLM Model, Chroma DB; Extract System Prompt; Define Chain")
        # 模型初始化
        MODEL = ChatOpenAI(
            base_url = base_url,
            api_key = api_key,
            model = chat_model,
            temperature = 0.8,
            top_p = 0.9,
            timeout = None,
            max_retries = 3
        )

        # embedding 模型初始化
        EMBEDDINGS = OpenAIEmbeddings(
            base_url = base_url,
            api_key = api_key,
            model = embedding_model,
            check_embedding_ctx_length = False
        )

        # 初始化ChromaDB对象
        VECTORSTORE = Chroma(
            persist_directory = chromadb_dir,
            collection_name = chromadb_collection,
            embedding_function = EMBEDDINGS
        )

        # 获取system_prompt
        system_prompt = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT)
        logger.info(f"system_prompt: {system_prompt}")
        SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个针对健康档案进行问答的机器人。你的任务是根据下述给定的已知信息回答用户问题。"),
                MessagesPlaceholder(variable_name = "history"),
                ("human",str(system_prompt.template))
            ]
        )

        # 定义chain
        # 将RAG检索放到LangChain的LCEL的chain中执行
        # 这段代码是使用Langchain框架中的`as_retriever`方法创建一个检索器对象
        # LangChain VectorStore对象不是 Runnable 的子类，因此无法集成到LangChain的LCEL的chain中
        # LangChain Retrievers是Runnable，实现了一组标准方法可集成到LCEL的chain中
        # `vectorstore`是一个向量存储对象，用于存储和检索文本数据
        # `as_retriever`方法将向量存储对象转换为一个检索器对象，该对象可以用于搜索与给定查询最相似的文本
        # `search_type`参数设置为"similarity"，表示使用相似度搜索算法
        # `search_kwargs`参数是一个字典，包含搜索算法的参数，这里的`k`参数设置为5，表示只返回与查询最相似的5个结果
        # retriever = VECTORSTORE.as_retriever(
        #     search_type = "similarity",
        #     search_kwargs = {"k": 5}
        # )

        CHAIN = SYSTEM_PROMPT | get_prompt | MODEL
        logger.info(f"Initializeing IS Done")

        #  处理带有消息历史Chain  将可运行的链与消息历史记录功能结合
        # RunnableWithMessageHistory允许在运行链时携带消息历史
        # 实例化的with_message_history是一个配置了消息历史的可运行对象，使用get_session_history来获取历史记录
        # ConfigurableFieldSpec定义了用户ID和会话ID的配置字段，使得这些字段在运行时可以被动态传递
        with_message_history = RunnableWithMessageHistory(
            CHAIN,
            get_session_history,
            input_messages_key = "query",
            history_messages_key = "history",
            history_factory_config = [
                ConfigurableFieldSpec(
                    id = "user_id",
                    annotation = str,
                    name = "USER ID",
                    description = "Unique identifier for the user.",
                    default = "",
                    is_shared =True
                ),
                ConfigurableFieldSpec(
                    id = "conversation_id",
                    annotation = str,
                    name = "CONVERSATION ID",
                    description = "Unique identifier for the conversation.",
                    default = "",
                    is_shared =True
                )
            ]
        )
    except Exception as e:
        logger.error(f"Error in initialization: {str(e)}")
        # raise 关键字重新抛出异常，以确保程序不会在错误状态下继续运行
        raise

    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield
    logger.info("Close Server Now...")

# lifespan 参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan = lifespan)


# POST请求接口，与大模型进行知识问答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not MODEL or not EMBEDDINGS or not VECTORSTORE or not CHAIN or not SYSTEM_PROMPT:
        logger.info("Server is not be initialized.")
        raise HTTPException(status_code = 500, detail = "Server is not initialized.")
    try:
        logger.info(f"Get Chat Request from User: {request}")
        user_query = request.messages[-1].content
        logger.info(f"User Query: {user_query}")

        retriever = VECTORSTORE.similarity_search(
            query = user_query,
            k = 3
        )

        # 调用CHAIN查询
        result = with_message_history.invoke(
            {
                "query": user_query,
                "context": retriever
            },
            config = {
                "configurable":{
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                }
            }
        )

        formatted_result = str(format_search_result(result.content))
        logger.info(f"Formatted Search Result: {formatted_result}")

        if request.stream:
            async def stream_generate():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                lines = formatted_result.split("\n")

                for idx, line in enumerate(lines):
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.comleiton.chunk",
                        "created": int(time.time()),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'},
                                # if i > 0 else {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }
                        ]
                    }

                    yield f"{json.dumps(chunk)}\n"

                    await asyncio.sleep(0.3)
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.comleiton.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delte": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"{json.dumps(final_chunk)}\n"
            return StreamingResponse(stream_generate(), media_type = "text/event-stream")
        else:
            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role = "assistant", content = formatted_result),
                        finish_reason="stop"
                    )
                ]
            )
            logger.info(f"发送响应内容: \n{response}")
            # 返回fastapi.responses中JSONResponse对象
            # model_dump()方法通常用于将Pydantic模型实例的内容转换为一个标准的Python字典，以便进行序列化
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Error in processing the Chat Session:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Start the Server in Port{PORT}")
    uvicorn.run(app, host="0.0.0.0", port = PORT)