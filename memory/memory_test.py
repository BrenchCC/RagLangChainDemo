# 用于记录日志，便于调试和追踪程序运行状态
import logging
# 用于与LLM的聊天模型交互
from langchain_openai import ChatOpenAI
# 配置可配置字段
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import PromptTemplate
# 定义聊天提示模板，以及占位符替换
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 用于运行带有消息历史的可运行对象
from langchain_core.runnables.history import RunnableWithMessageHistory
# 用于处理和存储对话历史
from langchain_community.chat_message_histories import SQLChatMessageHistory
import argparse

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='LLM对话系统')

    # 添加命令行参数
    parser.add_argument('--api_base', type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help='OpenAI API基础URL')
    parser.add_argument('--api_key', type=str, default="sk-1c33a0011b5c4ef6ac842d58255a20d8",
                        help='OpenAI API密钥')
    parser.add_argument('--model', type=str, default="qwen-plus",
                        help='OpenAI模型名称')
    parser.add_argument('--prompt_template', type=str, default="./memory/prompt_template.txt",
                        help='提示模板文件路径')
    parser.add_argument('--db_path', type=str, default="sqlite:///memory/memory.db",
                        help='SQLite数据库路径')
    parser.add_argument('--user_id', type=str, default="123",
                        help='用户ID')
    parser.add_argument('--conversation_id', type=str, default="123",
                        help='会话ID')
    parser.add_argument('--query', type=str, default="你好，我是NanGe!",
                        help='查询内容')
    parser.add_argument('--test', action='store_true',
                        help='运行测试对话')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数或默认值
    OPENAI_API_BASE = args.api_base
    OPENAI_CHAT_API_KEY = args.api_key
    OPENAI_CHAT_MODEL = args.model
    PROMPT_TEMPLATE_TXT = args.prompt_template
    DB_PATH = args.db_path
    USER_ID = args.user_id
    CONVERSATION_ID = args.conversation_id
    QUERY = args.query

    # 使用openai的model
    openai_model = ChatOpenAI(
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_CHAT_API_KEY,
        model=OPENAI_CHAT_MODEL,
    )

    # 定义prompt模版
    prompt_template = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", prompt_template.template),
            MessagesPlaceholder(variable_name="history"),
        ]
    )

    # 获取对话历史
    def get_session_history(user_id: str, conversation_id: str):
        return SQLChatMessageHistory(f"{user_id}--{conversation_id}", DB_PATH)

    # 获取prompt在chain中传递的prompt最终的内容
    def getPrompt(prompt):
        logger.info(f"最后给到LLM的prompt的内容: {prompt}")
        return prompt

    # 定义Chain
    chain = prompt | getPrompt | openai_model

    # 处理带有消息历史Chain
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="",
                is_shared=True,
            ),
        ],
    )

    if args.test:
        # 运行测试对话
        logger.info("运行测试对话...")

        # 第一次请求
        response1 = with_message_history.invoke(
            {"language": "中文", "query": "你好，我是Brench!"},
            config={"configurable": {"user_id": "123", "conversation_id": "123"}},
        )
        logger.info(f"response1: {response1.content}")

        # 第二次请求
        response2 = with_message_history.invoke(
            {"language": "中文", "query": "我叫什么?"},
            config={"configurable": {"user_id": "123", "conversation_id": "456"}},
        )
        logger.info(f"response2: {response2.content}")

        # 第三次请求
        response3 = with_message_history.invoke(
            {"language": "中文", "query": "我叫什么?"},
            config={"configurable": {"user_id": "456", "conversation_id": "123"}},
        )
        logger.info(f"response3: {response3.content}")

        # 第四次请求
        response4 = with_message_history.invoke(
            {"language": "中文", "query": "我叫什么?"},
            config={"configurable": {"user_id": "123", "conversation_id": "123"}},
        )
        logger.info(f"response4: {response4.content}")
    else:
        # 运行单次对话
        logger.info(f"运行单次对话: user_id={USER_ID}, conversation_id={CONVERSATION_ID}, query={QUERY}")

        response = with_message_history.invoke(
            {"language": "中文", "query": QUERY},
            config={"configurable": {"user_id": USER_ID, "conversation_id": CONVERSATION_ID}},
        )
        logger.info(f"响应: {response}")


if __name__ == "__main__":
    main()