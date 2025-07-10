import os
import re
import json
import asyncio
import uuid
import time
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
# prompt模版
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
# 部署REST API相关
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
# 向量数据库chroma相关
from langchain_chroma import Chroma
# openai的向量模型
from langchain_openai import OpenAIEmbeddings
# RAG相关
from langchain_core.runnables import RunnablePassthrough