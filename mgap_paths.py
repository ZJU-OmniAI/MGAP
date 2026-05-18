from pathlib import Path

MGAP_ROOT = Path(__file__).resolve().parent

def mgap_path(*parts: str) -> Path:
    return MGAP_ROOT.joinpath(*parts)
