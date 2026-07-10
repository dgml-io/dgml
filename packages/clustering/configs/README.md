# Example Hydra configs

These are **example** config-group files for the clustering framework. Each
maps to one section of the resolved :class:`clustering.config.schema.Config`
(see `src/clustering/config/schema.py`). They're intended as starting points —
copy and tune `model_id` / `embedding_dim` for your checkpoint.

## `encoder_image/`

Image-side encoder presets. The `# @package encoder_image` header at the top of
each file slots its contents into the `encoder_image` key, so you select one on
the command line:

```bash
# Qwen2.5-VL vision-tower embeddings
... encoder_image=qwen_vl

# Qwen3-VL-Embedding multimodal embeddings (MRL-truncatable dim); 8B or 2B
... encoder_image=qwen3_vl_embedding
... encoder_image=qwen3_vl_embedding_2b
```

`embedding_dim` **must** match the checkpoint you point `model_id` at — the
inline comments list the common values. A mismatch surfaces as a shape error in
the fusion module, not a silent bug.

To use the same preset on the text side, copy the file and change its header to
`# @package encoder_text` (text models such as E5/BGE additionally set
`doc_prefix` / `query_prefix`).
