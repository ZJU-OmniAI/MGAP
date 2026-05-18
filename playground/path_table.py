import os
from ._utils._colors import print_warning, print_note

from typing import List, Union
from pathlib import Path

_MGAP_ROOT = Path(__file__).resolve().parent.parent

_PATH_TABLE = {
    # You should set up these paths.
    # Please download COCO dataset from https://cocodataset.org/
    # `git clone https://github.com/junyangwang0410/AMBER.git`, then set up the AMBER first
    # `git clone https://github.com/Vision-CAIR/MiniGPT-4.git`, then set up the MiniGPT-4 first
    "llava-v1.5-7b": _MGAP_ROOT / "models" / "llava-v1.5-7b",
    "llava-v1.5-13b": _MGAP_ROOT / "models" / "llava-v1.5-13b",
    "Qwen-VL-Chat": "Qwen/Qwen-VL-Chat",
    "Qwen2-VL-7B-Instruct":  _MGAP_ROOT / "models" / "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-8B-Instruct":  _MGAP_ROOT / "models" / "Qwen3-VL-8B-Instruct",
    "COCO path": _MGAP_ROOT / "data" / "coco" / "val2014",
    "AMBER path": _MGAP_ROOT / "data" / "amber",
}

def get_path_from_table(name: str) -> Path:
    if name not in _PATH_TABLE:
        raise KeyError(f"'{name}' not found in path table.")
    path = _PATH_TABLE[name]
    print_note(f"Get '{name}' from path {path}")
    return Path(path)
