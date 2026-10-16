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
and stays loaded (about 1 GB, in VRAM or RAM, see below). The Models page sets an encoder per model (Vision button,
or the field in the add and pack forms); downloading a GGUF repository selects its `mmproj-*.gguf` (the f16 one when
there are several) and fills it in when you add or pack a model from the download. dsh needs `input: [text, image]` on the model's entry in `~/.dsh/settings.yaml`; its
`read_image` tool and image attachments then send images. In dsh's web UI an image is attached by dropping it on the
page or pasting it into the chat box (there is no attach button).

## Requests

OpenAI chat format, images as base64 data URLs or http(s) URLs (png, jpeg, bmp, gif):

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "What does the invoice say?"}]}
```

`{"type": "input_image", "image_url": "data:..."}` and Anthropic's `{"type": "image", "source": {"type": "base64",
"data": ...}}` (or `{"type": "url", "url": ...}`) work too. The server fetches http(s) URLs itself with curl (as the
Models page's downloads; up to 5 redirects, 20 MB, 20 s); a fetch that fails fails the request with curl's reason.
Anyone with the API key can make the server fetch a URL, including addresses on its own network.

## Encoding on another machine

On a 16 GB card the encoder runs on the CPU, which is slow for big images (a 1657x1085 screenshot: 1728 embeddings,
22 s). Another machine's GPU can encode instead; the encoder needs about 2 GB of VRAM.

1. On the serving machine, write a metadata-only copy of the base GGUF (10 MB; the encoder only reads the text model's
   metadata): `shoehorn encode-server --model base.gguf --save-meta base.meta.gguf`.
2. Copy it and the encoder file (mmproj) to the other machine, and start there:
   `shoehorn encode-server --mmproj mmproj-f16.gguf --model base.meta.gguf --host 10.77.0.2 --port 8091
   --api-key-file key.txt`. It encodes on its GPU when the encoder fits in the free VRAM (with 1.5 GB to spare), else on
   its CPU (`--cpu` forces that).
3. Start the server with `--encoder-url http://10.77.0.2:8091`. It sends each image's decoded pixels (width x height x 3
   bytes) and gets back the embeddings. The answer must have this end's token count and width, so a different encoder
   file is refused rather than misread. The server's API key goes along when it has one; the encoder checks it when
   started with `--api-key-file`.

When the remote encoder fails (off, unreachable, refused), the image is encoded locally as before and the remote is not
tried again for 60 s. The log's `image:` line says "on the remote encoder" when it encoded there.

## How it works

- **Encoder.** llama.cpp's mtmd library decodes, resizes and encodes the image. On the GPU when the card has the
  24 GB+ profile and the encoder (its file plus 1 GB of activations, plus 0.5 GB spare) fits in the free VRAM; the
  automatic residual budget (`--res-gpu-gb auto`) leaves that room when the model has an encoder. Otherwise on the CPU,
  as on 16 GB cards, where it does not fit next to the base. An encoder in VRAM is freed when the model unloads.
  `SHOEHORN_ENCODER=gpu` tries the GPU on any card (still only when it fits: ggml aborts on a failed allocation),
  `cpu` never. 640x400 image (260 embeddings): 1.95 s on the CPU (9800X3D), 0.03 s on the GPU (RTX 5080, a test with
  12 layers on the CPU to free VRAM; 0.4 s for the first image); about 5 s for 800x600 (475) on the CPU. The log's
  `image:` line gives the time and where it ran.
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
- 2026-10-15, remote encoder: the smoke test passes with every image encoded by an `encode-server` (on this machine's
  CPU, loaded from a metadata-only GGUF); with the encode server stopped, the next image falls back to the local
  encoder after the 2 s connection timeout and is read correctly.
- 2026-10-16, remote encoder on another machine: `encode-server` on a laptop RTX 4070 (8 GB, Windows 11 25H2), the
  server on the desktop, linked by gigabit Ethernet (through a monitor's USB hub, about 25-30 MB/s). Smoke test
  passes. A 640x400 image: 0.09-0.4 s on the laptop, 0.24-0.8 s with the transfer (desktop CPU: 2.0 s); a 1657x1085
  screenshot (1768 embeddings): 1.65 s on the laptop, 2.7 s with the transfer (5 MB of pixels out, 36 MB of embeddings
  back), against 22 s on the desktop CPU, and read correctly.
- 2026-10-15, follow-ups: smoke test passes with the encoder on the CPU and on the GPU; http and https URLs and
  Anthropic `url` sources read correctly, a 404 / refused connection / ftp URL give an error; an image dropped into
  dsh's web chat is read correctly, and so is one pasted into its chat box (a badge reading "Vision, still open"); the Models page selects a repository's mmproj, and add / pack set it on the model.
