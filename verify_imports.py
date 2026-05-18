from __future__ import annotations

import os
import sys

MGAP_ROOT = os.path.dirname(os.path.abspath(__file__))
if MGAP_ROOT not in sys.path:
    sys.path.insert(0, MGAP_ROOT)
os.chdir(MGAP_ROOT)

def check(name: str, fn) -> None:
    fn()
    print(f"[OK] {name}")

def main() -> None:
    check("torch", lambda: __import__("torch"))
    check("transformers", lambda: __import__("transformers"))
    check("llava", lambda: __import__("llava"))
    check("models_modified", lambda: __import__("models_modified"))
    check("decontext (import)", lambda: __import__("decontext"))

    import transformers
    from mgap_paths import mgap_path

    model_dir = mgap_path("models", "llava-v1.5-7b")
    if model_dir.exists():
        print(f"[OK] model dir exists: {model_dir}")
    else:
        print(f"[WARN] model dir missing: {model_dir} (place LLaVA weights before running)")

    print(f"\ntransformers: {transformers.__version__}")
    print(f"MGAP_ROOT: {MGAP_ROOT}")
    print("Import check passed.")

if __name__ == "__main__":
    main()
