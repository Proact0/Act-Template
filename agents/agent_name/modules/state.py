"""
상태 모듈 (필수수)

Langgraph State를 보관하는 모듈입니다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class State:
    """
    에이전트 상태 클래스

    에이전트의 상태를 저장하고 관리하는 클래스입니다.
    필요한 필드를 추가하여 에이전트의 상태를 정의할 수 있습니다.

    Attributes:
    """
