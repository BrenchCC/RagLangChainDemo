# pip install pdfminer.six

# 导入所需要的库
import logging

from markdown_it.rules_block import paragraph
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import re

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 处理中文文本，按照中文标点符号进行断句和语义分割
def sent_tokenize(input_string):
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 同时过滤空字符串
    return [sentence for sentence in sentences if sentence.strip()]

# 定义PDF文档处理函数，从PDF文件中按照指定的页码提取文字
def extract_text_from_pdf(filename, page_numbers, min_line_length):
    paragraphs = []
    buffer = ''
    full_text = ''

    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果制定了页码范围，跳过范围外的文件页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            # 将文件按照一行一行进行截取，并在每一行后面加上换行符
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    logger.info(f"full_text: {full_text}")

    # 按空行分隔，将文本组织为段落形式
    lines = full_text.split('\n')
    logger.info(f"lines length: {len(lines)}")

    # 将lines进行循环，取出每一个片段（text）进行处理合并成段落，处理逻辑为：
    # （1）首先判断text的最小行的长度是否大于min_line_length设置的值
    # （2）如果大于min_line_length，则将该text拼接在buffer后面，如果该text不是以连字符“-”结尾，则在行前加上一个空格；如果该text是以连字符“-”结尾，则去掉连字符）
    # （3）如果小于min_line_length且buffer中有内容，则将其添加到 paragraphs 列表中
    # （4）最后，处理剩余的缓冲区内容，在遍历结束后，如果 buffer 中仍有内容，则将其添加到 paragraphs 列表中
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    logger.info(f"paragraphs length: {len(paragraphs)}")

    return paragraphs

# 将PDF文档处理函数得到的文本列表再按一定粒度，部分重叠式的切割文本，使上下文更完整
# chunk_size：每个文本块的目标大小（以字符为单位），默认为 800
# overlap_size：块之间的重叠大小（以字符为单位），默认为 200
def split_text(paragraphs, chunk_size = 800, overlap_size = 200):
    # 按照指定 chunk_size 和 overlap_size 重叠割分文档
    sentences = [s.strip() for paragraph in paragraphs for s in sent_tokenize(paragraph)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev_len = 0
        prev = i - 1

        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' '
            prev -= 1
        chunk = overlap + chunk
        next = i + 1

        # 向后计算当前的chunk
        while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next

    logger.info(f"chunk size: {len(chunks)}")
    return chunks

def getParagraphs(filename, page_numbers, min_line_length):
    paragraphs = extract_text_from_pdf(filename, page_numbers, min_line_length)
    chunks = split_text(paragraphs, chunk_size = 800, overlap_size = 200)
    return chunks

if __name__ == "__main__":
    # 测试 PDF文档按一定条件处理成文本数据
    paragraphs = getParagraphs(
        "./input_pdf/健康档案.pdf",
        # page_numbers=[2, 3], # 指定页面
        page_numbers = None,  # 加载全部页面
        min_line_length = 1
    )
    # 测试前3条文本
    logger.info(f"只展示3段截取片段:")
    logger.info(f"截取的片段1: {paragraphs[0]}")
    logger.info(f"截取的片段2: {paragraphs[1]}")
    logger.info(f"截取的片段3: {paragraphs[2]}")