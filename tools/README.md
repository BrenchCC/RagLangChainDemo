# 文本预处理工具

在tools文件夹下提供了`pdfSplitTest_Ch.py`脚本工具用来处理中文文档、`pdfSplitTest_En.py`脚本工具用来处理英文文档。  
`vectordb_save.py`脚本执行调用tools中的工具进行文档预处理，进行向量计算并存储到向量数据库。

在使用命令`python vectordb_save.py`启动脚本时，可以通过命令行参数配置相关设置。以下是参数说明：

## 主要命令行参数

### 1. 模型设置相关
- `--api_base`：OpenAI API的基地址（URL），默认值为`https://api.wlai.vip/v1`
- `--api_key`：OpenAI API Key，默认值为`sk-t9pOWmiGVE02RBH88e87Eb8aE282471291F34640E787C2C6`
- `--embed_model`：嵌入模型名称，默认值为`text-embedding-3-small`

### 2. 语言与文件相关
- `--language`：文本语言，可选值为`Chinese`或`English`，默认值为`Chinese`
- `--input_pdf`：PDF文件路径，默认值为`input_pdf/健康档案.pdf`
- `--page_numbers`：指定处理的页码，用逗号分隔（如`2,3`表示处理第2页和第3页）。默认值为`None`，表示全部页码

### 3. 向量数据库相关
- `--chroma_dir`：向量数据库存储路径，默认值为`./chromaDB`
- `--collection`：向量集合名称，默认值为`demo001`

### 4. 查询参数（可选，用于测试）
- `--query_text`：测试查询文本，默认值为`张三九的基本信息是什么`（中文）或`llama2的安全性如何`（英文）
- `--query_num`：查询返回结果数量，默认值为`5`

## 使用示例

```bash
# 基本用法（使用默认值）
python vectordb_save.py

# 处理英文文档并测试查询
python vectordb_save.py \
  --language English \
  --input_pdf input_pdf/llama2.pdf \
  --page_numbers 2,3 \
  --query_text "llama2 safety features" \
  --query_num 3

# 自定义向量数据库存储路径和集合名称
python vectordb_save.py \
  --chroma_dir ./vector_db/chromaDB \
  --collection tech_docs \
  --input_pdf input_pdf/技术文档.pdf

# 使用自定义的OpenAI API地址和Key
python vectordb_save.py \
  --api_base https://api.example.com/v1 \
  --api_key your_api_key_here
注意事项
确保已经安装所有依赖库，包括openai、chromadb、pymupdf（用于处理PDF）等。
在处理大型PDF文件时，可能需要较长时间和较多的计算资源。
向量数据库的集合名称（--collection）用于区分不同的文档集合，相同集合的文档可以一起查询。
