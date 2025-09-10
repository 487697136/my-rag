"""
文本处理工具模块

本模块提供文本处理相关的工具函数，包括：
- 文本规范化
- 文本哈希计算
- 文本分割
- Token编码解码
- 字符串清理和转换
"""

import re
import json
import hashlib
import xxhash
from typing import List, Optional, Callable, Any


def normalize_text(text: str) -> str:
    """
    规范化文本，去除多余空白和特殊字符
    
    Args:
        text: 输入文本
        
    Returns:
        规范化后的文本
        
    Example:
        >>> normalize_text("  hello   world  ")
        'hello world'
    """
    if not text:
        return ""
        
    # 替换多个空白为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除前后空白
    text = text.strip()
    
    return text


def compute_text_hash(text: str, method: str = "xxh64") -> str:
    """
    计算文本哈希值
    
    Args:
        text: 输入文本
        method: 哈希方法，支持'xxh64'和'md5'
        
    Returns:
        哈希值字符串
        
    Example:
        >>> compute_text_hash("hello world")
        'a1b2c3d4e5f6...'
        >>> compute_text_hash("hello world", "md5")
        '5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    if not text:
        return ""
        
    # 规范化文本
    text = normalize_text(text)
    
    # 计算哈希值
    if method == "xxh64":
        # 使用xxhash（更快）
        return xxhash.xxh64(text.encode('utf-8')).hexdigest()
    else:
        # 使用md5（更通用）
        return hashlib.md5(text.encode('utf-8')).hexdigest()


def split_text_by_length(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    按长度分割文本
    
    Args:
        text: 输入文本
        max_length: 最大长度
        overlap: 重叠长度
        
    Returns:
        分割后的文本列表
        
    Example:
        >>> split_text_by_length("long text...", max_length=100, overlap=20)
        ['chunk1', 'chunk2', ...]
    """
    if not text:
        return []
        
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # 确定当前块的结束位置
        end = start + max_length
        
        if end >= len(text):
            # 如果到达文本末尾，直接添加剩余部分
            chunks.append(text[start:])
            break
            
        # 尝试在适当位置分割（句号、问号、感叹号、换行符）
        split_pos = max(
            text.rfind('. ', start, end),
            text.rfind('? ', start, end),
            text.rfind('! ', start, end),
            text.rfind('\n', start, end)
        )
        
        if split_pos > start:
            # 找到合适的分割点
            chunks.append(text[start:split_pos+1])
            start = split_pos + 1
        else:
            # 没有找到合适的分割点，强制分割
            chunks.append(text[start:end])
            start = end - overlap  # 保留一定重叠
            
    return chunks


def clean_str(text: str) -> str:
    """
    清理字符串，去除前后空白
    
    Args:
        text: 输入字符串
        
    Returns:
        清理后的字符串
        
    Example:
        >>> clean_str("  hello world  ")
        'hello world'
    """
    if not text:
        return ""
    return text.strip()


def decode_tokens_by_tiktoken(tokens: List[int], tiktoken_model) -> str:
    """
    使用tiktoken解码token列表为字符串
    
    Args:
        tokens: token列表
        tiktoken_model: tiktoken模型
        
    Returns:
        解码后的字符串
        
    Example:
        >>> decode_tokens_by_tiktoken([123, 456], tiktoken_model)
        'decoded text'
    """
    return tiktoken_model.decode(tokens)


def encode_string_by_tiktoken(text: str, tiktoken_model) -> List[int]:
    """
    使用tiktoken编码字符串为token列表
    
    Args:
        text: 输入字符串
        tiktoken_model: tiktoken模型
        
    Returns:
        token列表
        
    Example:
        >>> encode_string_by_tiktoken("hello world", tiktoken_model)
        [123, 456, 789]
    """
    return tiktoken_model.encode(text)


def is_float_regex(text: str) -> bool:
    """
    检查字符串是否为浮点数
    
    Args:
        text: 输入字符串
        
    Returns:
        是否为浮点数
        
    Example:
        >>> is_float_regex("123.45")
        True
        >>> is_float_regex("abc")
        False
    """
    return bool(re.match(r'^-?\d*\.?\d+$', text))


def list_of_list_to_csv(data: List[List[Any]], delimiter: str = ',') -> str:
    """
    将二维列表转换为CSV字符串
    
    Args:
        data: 二维列表
        delimiter: 分隔符
        
    Returns:
        CSV字符串
        
    Example:
        >>> list_of_list_to_csv([['a', 'b'], ['c', 'd']])
        'a,b\nc,d'
    """
    return '\n'.join([delimiter.join(map(str, row)) for row in data])


def pack_user_ass_to_openai_messages(user_message: str, assistant_message: Optional[str] = None) -> List[dict]:
    """
    将用户和助手消息打包为OpenAI格式
    
    Args:
        user_message: 用户消息
        assistant_message: 助手消息，可选
        
    Returns:
        OpenAI格式的消息列表
        
    Example:
        >>> pack_user_ass_to_openai_messages("hello", "hi there")
        [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi there'}]
    """
    messages = [{"role": "user", "content": user_message}]
    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message})
    return messages


def split_string_by_multi_markers(text: str, markers: List[str]) -> List[str]:
    """
    使用多个标记分割字符串
    
    Args:
        text: 输入字符串
        markers: 标记列表
        
    Returns:
        分割后的字符串列表
        
    Example:
        >>> split_string_by_multi_markers("a.b,c;d", ['.', ',', ';'])
        ['a', 'b', 'c', 'd']
    """
    pattern = '|'.join(map(re.escape, markers))
    return re.split(pattern, text)


def truncate_list_by_token_size(items: List[Any], max_tokens: int, tiktoken_model, key: Optional[Callable] = None) -> List[Any]:
    """
    根据token数量截断列表
    
    Args:
        items: 项目列表
        max_tokens: 最大token数量
        tiktoken_model: tiktoken模型
        key: 用于获取文本的函数，默认为None
        
    Returns:
        截断后的列表
        
    Example:
        >>> truncate_list_by_token_size(['text1', 'text2'], 100, tiktoken_model)
        ['text1']  # 如果text1的token数已接近100
    """
    total_tokens = 0
    result = []
    for item in items:
        # 如果提供了key函数，使用key函数获取要编码的文本
        text_to_encode = key(item) if key else str(item)
        item_tokens = len(tiktoken_model.encode(text_to_encode))
        if total_tokens + item_tokens <= max_tokens:
            result.append(item)
            total_tokens += item_tokens
        else:
            break
    return result


def convert_response_to_json(response: str) -> dict:
    """
    将响应转换为JSON格式
    
    Args:
        response: 响应字符串
        
    Returns:
        JSON对象
        
    Example:
        >>> convert_response_to_json('{"key": "value"}')
        {'key': 'value'}
        >>> convert_response_to_json('plain text')
        {'response': 'plain text'}
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 如果解析失败，返回原始响应
        return {"response": response} 