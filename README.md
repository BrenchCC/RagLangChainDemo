# RagLangChainDemo

# 项目简介
在本项目中模拟健康档案私有知识库构建和检索全流程，通过一份代码实现了同时支持多种大模型（如OpenAI、doubao、阿里通义千问等）的RAG（检索增强生成）功能:
- (1)离线步骤:文档加载->文档切分->向量化->灌入向量数据库；
- (2)在线步骤:获取用户问题->用户问题向量化->检索向量数据库->将检索结果和用户问题填入prompt模版->用最终的prompt调用LLM->由LLM生成回复

# 1.基础概念
## 1.1 RAG定义及技术方案架构
### （1）RAG定义
RAG:Retrieval Augmented Generation(检索增强生成):通过使用检索的方法来增强生成模型的能力       
核心思想:人找知识，会查资料；LLM找知识，会查向量数据库(如果是Agent构建，可能还会加入内部的搜索工具)        
主要目标:补充LLM固有局限性，LLM的知识不是实时的，LLM可能不知道私有领域业务知识          
场景类比:可以把RAG的过程想象成为开卷考试。让LLM先翻书查找相关答案，再回答问题              
### （2）技术方案架构
离线步骤:文档加载->文档切分->向量化->灌入向量数据库           
在线步骤:获取用户问题->用户问题向量化->检索向量数据库->将检索结果和用户问题填入prompt模版->用最终的prompt调用LLM->由LLM生成回复             
### （3）几个关键概念：
向量数据库的意义是快速的检索             
向量数据库本身不生成向量，向量是由Embedding模型产生的             
向量数据库与传统的关系型数据库是互补的，不是替代关系，在实际应用中根据实际需求经常同时使用               

## 1.2 LangChain
### （1）LangChain定义
LangChain是一个用于开发由大型语言模型(LLM)驱动的应用程序的框架，官方网址：https://python.langchain.com/v0.2/docs/introduction/          
### （2）LCEL定义
LCEL(LangChain Expression Language),原来叫chain，是一种申明式语言，可轻松组合不同的调用顺序构成chain            
其特点包括流支持、异步支持、优化的并行执行、重试和回退、访问中间结果、输入和输出模式、无缝LangSmith跟踪集成、无缝LangServe部署集成            
### （3）LangSmith
LangSmith是一个用于构建生产级LLM应用程序的平台。通过它，您可以密切监控和评估您的应用程序，官方网址：https://docs.smith.langchain.com/          

## 1.3 Chroma
向量数据库，专门为向量检索设计的中间件      

# 2.项目初始化
## 2.1 下载源码
## 2.2 构建项目
使用pycharm构建一个项目，为项目配置虚拟python环境               
项目名称：RagLangChainDemo                 

## 2.3 将相关代码拷贝到项目工程中           
直接将下载的文件夹中的文件拷贝到新建的项目目录中               

## 2.4 安装项目依赖          
pip install -r requirements.txt            
每个软件包后面都指定了本次视频测试中固定的版本号  


# 3.项目测试
## 3.1 准备测试文档
这里以pdf文件为例，在input_pdf文件夹下准备了4份pdf文件:                
健康档案.pdf:测试中文pdf文档处理                
llama2.pdf:测试英文pdf文档处理
两个带表格的pdf

## 3.2 文本预处理后进行灌库
在tools文件夹下提供了`pdfSplitTest_Ch.py`脚本工具用来处理中文文档、`pdfSplitTest_En.py`脚本工具用来处理英文文档。  
`vectordb_save.py`脚本执行调用tools中的工具进行文档预处理，进行向量计算并存储到向量数据库。

在使用命令`python vectordb_save.py`启动脚本时，可以通过命令行参数配置相关设置。
详情见[TOOLS_README](./tools/README.md)

## 3.3 启动服务器server
在使用`python main.py`命令启动脚本前，需根据自己的实际情况调整代码中的参数，具体参数参考代码

## 3.4 启动测试脚本测试sever服务
在运行`python apiTest.py`命令启动脚本前，需根据自己的实际情况调整代码中的如下参数，运行成功后，可以查看smith的跟踪情况                  
**调整1:默认非流式输出 True or False**                  
stream_flag = False                  
**调整2:检查URL地址中的IP和PORT是否和main脚本中相同**                  
url = "http://localhost:8081/v1/chat/completions"