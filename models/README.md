# Model weights

Place **LLaVA-1.5-7B** weights in:

```
models/llava-v1.5-7b/
```

The directory should contain HuggingFace-style files (`config.json`, weights, tokenizer, etc.).

Or specify a custom path at runtime:

```bash
python decontext.py --model-path /path/to/llava-v1.5-7b ...
```
