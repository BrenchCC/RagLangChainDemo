# 文本预处理工具

在tools文件夹下提供了pdfSplitTest_Ch.py脚本工具用来处理中文文档、pdfSplitTest_En.py脚本工具用来处理英文文档                
vectorSaveTest.py脚本执行调用tools中的工具进行文档预处理后进行向量计算及灌库                
在使用python vectorSaveTest.py命令启动脚本前，需根据自己的实际情况调整代码中的如下参数：             
**调整1:选择使用哪种模型标志设置:**              
API_TYPE = "oneapi"  # openai:调用gpt模型；oneapi:调用oneapi方案支持的模型(这里调用通义千问)               
**调整2:openai模型相关配置 根据自己的实际情况进行调整:**                
OPENAI_API_BASE = "这里填写API调用的URL地址"               
OPENAI_EMBEDDING_API_KEY = "这里填写Embedding模型的API_KEY"              
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"             
**调整3:oneapi相关配置(通义千问为例) 根据自己的实际情况进行调整:**             
ONEAPI_API_BASE = "这里填写oneapi调用的URL地址"            
ONEAPI_EMBEDDING_API_KEY = "这里填写Embedding模型的API_KEY"                
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"                    
**调整4:配置测试文本类型:**            
TEXT_LANGUAGE = 'Chinese'  #Chinese 或 English                 
**调整5:配置待处理的PDF文件路径:**               
INPUT_PDF = "input_pdf/健康档案.pdf"              
**调整6:指定文件中待处理的页数范围，全部页数则填None:**               
PAGE_NUMBERS=None                  
PAGE_NUMBERS=[2, 3] # 指定页数     
**调整7:设置向量数据库chromaDB相关:**               
CHROMADB_DIRECTORY = "chromaDB"  # chromaDB向量数据库的持久化路径             
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称 
