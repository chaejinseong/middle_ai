"""실행 장치 선택: CUDA(NVIDIA) → MPS(Apple Silicon) → CPU"""
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str):
    # MPS는 bf16 지원이 버전마다 달라 fp16/fp32를 씀
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32
