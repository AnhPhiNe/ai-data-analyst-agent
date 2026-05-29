import json
import pandas as pd
from backend.agent.column_resolver import resolve_column
from backend.agent.gemini_runtime import LLMProvider


class MockProvider(LLMProvider):
    def __init__(self, response_dict: dict):
        self.response_dict = response_dict
        
    def generate_structured(self, prompt: str) -> str:
        return json.dumps(self.response_dict)
        
    def generate_text(self, prompt: str) -> str:
        return ""


def test_resolve_column_with_llm() -> None:
    df = pd.DataFrame(
        {
            "department": ["HR", "Sales"],
            "Monthly_Revenue": [1000.0, 1500.0],
            "is_active": [True, False],
        }
    )

    # 1. Exact match mock
    provider1 = MockProvider({"matched_column": "Monthly_Revenue"})
    assert resolve_column(df, "doanh thu hang thang", provider=provider1) == "Monthly_Revenue"

    # 2. Type filtered mock (Categorical)
    provider2 = MockProvider({"matched_column": "department"})
    assert resolve_column(df, "phong ban", provider=provider2, expected_type="categorical") == "department"

    # 3. Ambiguous/Null mock
    provider3 = MockProvider({"matched_column": None})
    assert resolve_column(df, "khong ro rang", provider=provider3) is None

def test_resolve_column_handles_exception() -> None:
    class ErrorProvider(LLMProvider):
        def generate_structured(self, prompt: str) -> str:
            raise ValueError("API Error")
            
        def generate_text(self, prompt: str) -> str:
            return ""
            
    df = pd.DataFrame({"col1": [1, 2]})
    assert resolve_column(df, "text", provider=ErrorProvider()) is None
