# Images (vision)

The 27B (`Qwen3_5ForConditionalGeneration`) has a vision encoder: 27 layers, about 0.45B parameters, output width
5120 (the text model's). shoehorn's packs leave it out; it runs from llama.cpp's encoder file (an "mmproj" GGUF,
f16, 0.93 GB), for example `mmproj-Qwen3.8-27B-Uncensored-f16.gguf` from orcarouter's GGUF repository.

## Setup

The server finds a model's encoder in this order:

1. `"mmproj": "path"` on the model's entry in the models list (`~/.shoehorn/models.json`);
2. `serve --mmproj enc.gguf` for the main model;
3. a single `mmproj*.gguf` in the model's folder.

Without an encoder a request with an image fails with an error that says so. The encoder loads on the first image
and stays in RAM (about 1 GB). dsh needs `input: [text, image]` on the model's entry in `~/.dsh/settings.yaml`; its
`read_image` tool and image attachments then send images.

## Requests

OpenAI chat format, images as base64 data URLs (png, jpeg, bmp, gif):

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "What does the invoice say?"}]}
```

`{"type": "input_image", "image_url": "data:..."}` and Anthropic's `{"type": "image", "source": {"type": "base64",
"data": ...}}` work too. http(s) URLs are not fetched.

## How it works

- **Encoder.** llama.cpp's mtmd library decodes, resizes and encodes the image on the CPU (it does not fit in VRAM
  next to a 16 GB base): 2-3 s for a 640x400 image (260 embeddings), about 5 s for 800x600 (475), on the 9800X3D.
- **Prompt.** An image is a run of placeholder tokens in the prompt, one per embedding, between the template's
  `<|vision_start|>` and `<|vision_end|>`. Their ids come from a hash of the image's pixels, so prompt reuse (which
  compares token ids) reuses a prefix only when its images are the same. A follow-up turn about the same image reuses
  it without encoding it again.
- **Model.** `Qwen35::eval` takes the embeddings in place of token embeddings (`EvalOpts::embd`) with M-RoPE positions
  (`EvalOpts::pos`: sections t, y, x): the image's rows sit on its 2D grid. An image of n embeddings on an h x w grid
  takes max(h, w) rotary positions, not n, so text after it continues at sequence position minus a rope shift
  (`set_rope_shift`). The KV cache stays indexed by sequence position, so commit, rollback, slots, the side sequence
  and drafting are unchanged; each sequence (slot, side sequence, draft member) carries its own shift.
- **Decoding** is the usual speculative path; images only lengthen the prompt.

## Checks

- `shoehorn vision <base.gguf> [--res r.gguf] --mmproj enc.gguf --image a.png [--image b.png] -p "..."` answers one
  prompt with plain decoding. With the base alone it reproduces llama.cpp's `llama-mtmd-cli` answer word for word
  (same model and encoder, greedy). `--layout index` (no rope shift) gives a different answer, so the shift matters.
- `tests/vision_smoke.py --url ... --image a.png --image2 b.png --image3 c.png --expect ... --expect3 ...`: one
  image, a follow-up turn (reuses the prompt), a same-size image with the same question and a different image in the
  same place (neither may reuse the first one's state), two images in one message, streaming.
- 2026-10-15, 5080, orca 27B (Q4_K residual): all pass; an image plus tools gives a correct tool call; dsh's
  `read_image` reads an invoice correctly.
