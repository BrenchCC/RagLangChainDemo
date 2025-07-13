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