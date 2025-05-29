"""
LLM 모델 설정 모듈 (선택)

이 모듈은 LangChain에서 사용할 LLM 모델을 설정하고 반환하는 함수들을 포함합니다.
"""

import os
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(
    model_name: str = "gpt-3.5-turbo", temperature: float = 0.7
) -> BaseChatModel:
    """
    LLM 모델을 가져오는 함수

    환경 변수에서 API 키를 가져와서 LLM 모델을 초기화하고 반환합니다.

    Args:
        model_name (str, optional): 사용할 모델 이름. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): 온도 설정. Defaults to 0.7.

    Returns:
        BaseChatModel: 초기화된 LLM 모델
    """
    # 환경 변수에서 API 키 가져오기
    api_key = os.environ.get("OPENAI_API_KEY")

    # LLM 모델 초기화
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    return llm
