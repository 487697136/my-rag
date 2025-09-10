"""
文本分割器模块

本模块提供基于分隔符的文本分割功能，主要用于将长文本按照指定的分隔符进行智能分割。
"""

from typing import List, Optional, Union, Literal, Callable


class SeparatorSplitter:
    """
    基于分隔符的文本分割器
    
    该类提供了基于指定分隔符对token序列进行智能分割的功能，
    支持重叠分割和长度控制。
    
    Attributes:
        _separators: 分隔符列表，每个分隔符是一个token序列
        _keep_separator: 分隔符保留策略
        _chunk_size: 最大块大小
        _chunk_overlap: 块间重叠大小
        _length_function: 长度计算函数
    """
    
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[List[int]], int] = len,
    ):
        """
        初始化分割器
        
        Args:
            separators: 分隔符列表，每个分隔符是一个token序列
            keep_separator: 分隔符保留策略
                - True: 保留分隔符到当前块末尾
                - "start": 保留分隔符到下一个块开头
                - "end": 保留分隔符到当前块末尾
            chunk_size: 最大块大小（token数量）
            chunk_overlap: 块间重叠大小（token数量）
            length_function: 长度计算函数，默认为len
            
        Example:
            >>> splitter = SeparatorSplitter(
            ...     separators=[[13, 10], [46]],  # 换行符和句号
            ...     chunk_size=1000,
            ...     chunk_overlap=100
            ... )
        """
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        """
        分割token序列
        
        Args:
            tokens: 要分割的token序列
            
        Returns:
            分割后的token块列表
            
        Example:
            >>> tokens = [1, 2, 3, 13, 10, 4, 5, 6]  # 包含换行符
            >>> chunks = splitter.split_tokens(tokens)
            >>> len(chunks)  # 应该被分割成多个块
            2
        """
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        """
        使用分隔符分割token序列
        
        Args:
            tokens: 要分割的token序列
            
        Returns:
            按分隔符分割的token块列表
        """
        splits = []
        current_split = []
        i = 0
        
        while i < len(tokens):
            separator_found = False
            
            # 检查是否匹配任何分隔符
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    # 根据保留策略处理分隔符
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    
                    # 保存当前分割块
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    
                    # 如果分隔符保留到下一个块开头
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    
                    i += len(separator)
                    separator_found = True
                    break
            
            # 如果没有找到分隔符，继续添加token
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        
        # 添加最后一个分割块
        if current_split:
            splits.append(current_split)
        
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        """
        合并分割块，确保不超过最大块大小
        
        Args:
            splits: 分割后的token块列表
            
        Returns:
            合并后的token块列表
        """
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                # 第一个块直接添加
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                # 如果合并后不超过最大大小，则合并
                current_chunk.extend(split)
            else:
                # 否则保存当前块，开始新块
                merged_splits.append(current_chunk)
                current_chunk = split

        # 添加最后一个块
        if current_chunk:
            merged_splits.append(current_chunk)

        # 如果只有一个块且超过最大大小，强制分割
        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        # 应用重叠策略
        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        """
        强制分割超过最大大小的块
        
        Args:
            chunk: 要分割的token块
            
        Returns:
            分割后的token块列表
        """
        result = []
        step_size = self._chunk_size - self._chunk_overlap
        
        for i in range(0, len(chunk), step_size):
            new_chunk = chunk[i:i + self._chunk_size]
            # 只有当块长度大于重叠大小时才添加
            if len(new_chunk) > self._chunk_overlap:
                result.append(new_chunk)
        
        return result

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        """
        在块之间应用重叠策略
        
        Args:
            chunks: 要应用重叠的token块列表
            
        Returns:
            应用重叠后的token块列表
        """
        result = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 第一个块直接添加
                result.append(chunk)
            else:
                # 后续块添加前一个块的重叠部分
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                
                # 确保不超过最大块大小
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                
                result.append(new_chunk)
        
        return result 