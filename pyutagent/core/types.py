"""类型定义 - 完善类型系统"""
from typing import NewType, TypedDict, List, Optional, NotRequired


# NewType 定义
FilePath = NewType('FilePath', str)
ClassName = NewType('ClassName', str)
MethodName = NewType('MethodName', str)


class CoveragePercentage(float):
    """覆盖率百分比（0.0-1.0）"""
    
    def __new__(cls, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Coverage must be between 0.0 and 1.0, got {value}")
        return super().__new__(cls, value)


# TypedDict 定义
class CompilationResultDict(TypedDict):
    """编译结果"""
    success: bool
    output: str
    errors: List[str]
    duration: float
    is_incremental: NotRequired[bool]


class TestResultDict(TypedDict):
    """测试结果"""
    total: int
    passed: int
    failed: int
    errors: int
    duration: float


class ComponentInfo(TypedDict):
    """组件信息"""
    name: str
    type: str
    version: str
    description: NotRequired[str]
    enabled: NotRequired[bool]


class AgentResultDict(TypedDict):
    """Agent 结果"""
    success: bool
    coverage: float
    iterations: int
    test_file: str
    errors: List[str]
