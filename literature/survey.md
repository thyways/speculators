# Local reference survey

- `examples/train/dflash_qwen3_6_35b_a3b_perfectblend_online_full.sh`: local canonical example for AdamW, 4% warmup, cosine decay, and bidirectional DFlash attention.
- `/inspire/sfs/project/inf-multimodal/public/wumengke/HunyuanOCR/train/dflash_draft.py`: confirmed working VLM DFlash implementation whose draft attention is explicitly non-causal (`is_causal = False`).
- `/inspire/sfs/project/inf-multimodal/public/wumengke/HunyuanOCR/docs/training_zh.md`: documents the working multimodal DFlash training/data pipeline and approximately one-million-sample from-scratch regime.
