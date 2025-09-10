"""
答案生成模块

提供与LLM交互生成答案相关的功能
"""

# 导入提示模板相关功能
try:
    from .prompts import PromptTemplate, ConfidenceAwarePrompt, BasicPromptTemplate, MultiHopPromptTemplate, PromptLibrary
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    PromptTemplate = None
    ConfidenceAwarePrompt = None
    BasicPromptTemplate = None
    MultiHopPromptTemplate = None
    PromptLibrary = None

# generator模块暂时不可用，可以在未来添加
# from .generator import AnswerGenerator, AnswerGeneratorConfig

__all__ = [
    # 'AnswerGenerator',        # 暂时不可用
    # 'AnswerGeneratorConfig',  # 暂时不可用
    'PromptTemplate',
    'BasicPromptTemplate',
    'ConfidenceAwarePrompt',
    'MultiHopPromptTemplate', 
    'PromptLibrary',
] 