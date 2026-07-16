import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: tests that require a CUDA GPU")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return torch.cuda.is_available()
