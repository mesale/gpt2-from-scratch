import pytest
from src.tokenizer import load_tokenizer

@pytest.fixture(scope="session")
def tok():
    return load_tokenizer("configs/tokenizer.json")