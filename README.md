# 🍎 mac code

**Open-source AI agent that runs powerful models on your Mac — leveraging Apple Silicon like it was meant to be used.**

There are 100 million Macs with Apple Silicon in the world. Every one of them has a GPU, unified memory, and a fast SSD. That's the largest untapped AI compute network on Earth — and it's sitting on people's desks doing nothing.

mac code is an open-source project to change that. We run frontier-class open-weight models locally on Macs, with no cloud dependency, no API keys, and no cost per token. The goal is to make every Mac an AI workstation.

---

## The Mission

**Put open-source AI models on every Mac.**

Apple Silicon is uniquely suited for local LLM inference. Unlike NVIDIA GPUs, Apple's unified memory architecture lets the CPU, GPU, and SSD share the same address space. This means:

- Models that don't fit in RAM can **page from the SSD** and the GPU still processes everything
- No PCIe bottleneck, no CPU fallback, no data copies between devices
- A $600 Mac mini can run a 35-billion parameter model at 30 tokens/second

We proved this works. Now we're open-sourcing the entire stack so anyone with a Mac can do it too.

## Two models, one agent

mac code runs two models and **automatically switches between them** based on your query:

| Model | When it's used | Speed | Context | Why |
|---|---|---|---|---|
| **Qwen3.5-9B** (Q4_K_M, 5.3 GB) | Web search, tools, file ops, agent tasks | 30 tok/s | **32K** | Reliable tool calling at 5.1 bits/weight |
| **Qwen3.5-35B-A3B** (IQ2_M, 10.6 GB) | Reasoning, math, knowledge, fast answers | 57 tok/s | 8K | Smarter model, MoE architecture |

The agent detects your intent:
- *"When do the Lakers play?"* → routes to **9B** (needs web search)
- *"Explain quantum computing"* → routes to **35B** (pure reasoning, faster)

The model swap takes ~20 seconds. You can also switch manually with `/model 9b` or `/model 35b`.

## What makes this different

**The 35B model doesn't fit in RAM.** That's the whole point.

Qwen3.5-35B-A3B is a 10.6 GB model. A Mac mini M4 has 16 GB of RAM. After macOS takes its share, there's not enough room. The overflow pages from the SSD.

On any other hardware, this kills performance. We tested it:

| Setup | Speed | Cost/hr | What happens |
|---|---|---|---|
| **Mac mini M4 + SSD paging** | **29.8 tok/s** | **$0.00** | **GPU processes everything via unified memory** |
| NVIDIA GPU + NVMe paging | 1.6 tok/s | $0.44 | CPU bottleneck — GPU can't access paged data |
| NVIDIA GPU + FUSE paging | 0.075 tok/s | $0.44 | Network storage — barely functional |
| NVIDIA GPU in-VRAM (no paging) | 42.5 tok/s | $0.34 | Fast, but costs money and needs big GPU |
| Claude Code (API) | ~80 tok/s | ~$0.50+ | Fastest, but every token costs money |

**Apple Silicon is 18.6x faster than NVIDIA when the model doesn't fit in memory.**

Why? **Unified memory.** On a Mac, the CPU, GPU, and SSD share the same memory address space. When macOS pages model weights from the SSD, the Metal GPU still processes them directly — no PCIe bottleneck. On NVIDIA, paging forces data through the CPU first and the GPU sits idle.

**This is Apple's "LLM in a Flash" thesis running in practice on a $600 computer.**

---

## Quick Start

### What you need

- Mac with Apple Silicon (M1 or later, 16GB+ RAM)
- [Homebrew](https://brew.sh)

### One-command setup

```bash
git clone https://github.com/walter-grace/mac-code.git
cd mac-code
chmod +x setup.sh && ./setup.sh
```

### Or step by step

**1 — Install dependencies**

```bash
brew install llama.cpp go
pip3 install huggingface-hub rich --break-system-packages
```

**2 — Download both models**

```bash
mkdir -p ~/models

# 9B model (recommended — tools + agent work reliably)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

# 35B MoE model (optional — faster reasoning, SSD flash-paging)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

**3 — Build PicoClaw (agent framework)**

```bash
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
```

**4 — Configure**

```bash
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json
```

**5 — Start the server and run**

```bash
# Start with the 9B model (recommended)
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 32768 \
    --n-gpu-layers 99 --reasoning off -t 4

# In another terminal
python3 agent.py
```

The agent will auto-swap to the 35B model when it detects a reasoning-only question (if the 35B model is downloaded).

---

## What it looks like

```
  🍎 mac code
  claude code, but it runs on your Mac for free

  model  Qwen3.5-9B  8.95B dense · Q4_K_M · 32K ctx
  tools  search · fetch · exec · files
  cost   $0.00/hr  Apple M4 Metal · localhost:8000

  type / to see all commands

  auto 9b > when do the lakers play next?

  ⠹ searching  8s
    tool_call web_search "Lakers next game"
    tool_result received

  The Lakers play tonight at 7 PM ET against the Detroit Pistons.

  29.7 tok/s  ·  42 tokens  ·  1.4s

  auto 9b > explain how backpropagation works

  routing to 35B (faster reasoning)...
  ⠹ swapping to 35B  18s
  Switched to Qwen3.5-35B-A3B (8K ctx)

  Backpropagation is the fundamental algorithm for training
  neural networks...

  57.1 tok/s  ·  200 tokens  ·  3.5s
```

## Commands

Type `/` to see all 20 commands:

| Command | Action |
|---|---|
| `/agent` | Agent mode — tools + web search (default) |
| `/raw` | Raw mode — direct streaming, no tools |
| `/model 9b` | Switch to 9B (32K ctx, tool calling) |
| `/model 35b` | Switch to 35B MoE (8K ctx, faster) |
| `/auto` | Toggle smart auto-routing between models |
| `/btw <q>` | Side question without adding to conversation |
| `/search <q>` | Quick web search |
| `/bench` | Speed benchmark |
| `/loop 5m <p>` | Run prompt on recurring interval |
| `/stop` | Stop a running loop |
| `/branch` | Save conversation checkpoint |
| `/restore` | Restore to checkpoint |
| `/add-dir <path>` | Set working directory |
| `/save <file>` | Export conversation to JSON |
| `/clear` | Reset conversation |
| `/stats` | Session statistics |
| `/tools` | List available tools |
| `/system <msg>` | Set system prompt |
| `/compact` | Toggle markdown rendering |
| `/cost` | Show cost savings vs cloud |
| `/quit` | Exit |

## Tools

All local. No API keys needed.

| Tool | Source |
|---|---|
| `web_search` | DuckDuckGo |
| `web_fetch` | Read any URL |
| `exec` | Shell commands |
| `read_file` | Local files |
| `write_file` | Create files |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  mac code TUI              Python + Rich             │
│  Smart routing: tools → 9B, reasoning → 35B          │
├──────────────────────────────────────────────────────┤
│  PicoClaw                  Go agent framework        │
│  github.com/sipeed/picoclaw                          │
├──────────────────────────────────────────────────────┤
│  llama.cpp                 C++ + Metal GPU           │
│  OpenAI-compatible API @ localhost:8000              │
├──────────────────────────────────────────────────────┤
│  Qwen3.5-9B (Q4_K_M)      Qwen3.5-35B-A3B (IQ2_M)  │
│  32K ctx, tools            8K ctx, fast reasoning    │
├──────────────────────────────────────────────────────┤
│  Apple Silicon             Unified Memory + SSD      │
│  Metal GPU + flash paging                            │
└──────────────────────────────────────────────────────┘
```

### Why two models?

The 35B MoE model is smarter and faster (57 tok/s) but its aggressive 2.6-bit quantization breaks tool-calling — the model loops on tool calls without answering. The 9B model at 5.1 bits/weight follows instructions reliably and has 32K context for multi-step agent tasks.

Auto-routing gives you the best of both: tool reliability from the 9B, and raw intelligence from the 35B.

---

## Scaling

| Mac | RAM | What you can run | Est. Speed |
|---|---|---|---|
| Any Mac (8GB) | 8 GB | 9B only, 4K context | ~15 tok/s |
| **Mac mini M4** | **16 GB** | **9B (32K ctx) + 35B MoE (8K ctx, SSD paging)** | **30-57 tok/s** |
| Mac mini M4 Pro | 48 GB | 35B MoE at Q4_K_M, 32K context | ~40+ tok/s |
| Mac Studio M4 Max | 128 GB | Qwen3.5-397B-A17B (frontier MoE) | ~10-20 tok/s |
| Mac Studio M4 Ultra | 192 GB | Qwen3.5-397B-A17B at Q4_K_M | ~15-30 tok/s |
| Mac Pro M4 Ultra | 512 GB | **Kimi K2.5 (1T MoE, 32B active)** | ~5-15 tok/s |

### The 1T model frontier

Models like **Kimi K2.5** (1 trillion parameters, MoE with 32B active per token) currently require cloud GPUs with 8x H100s (~$25/hr). But the same MoE + SSD paging principle applies: a Mac Pro with 512 GB unified memory could potentially run a Q2 quantized 1T model at ~300 GB on disk, with the GPU processing all layers via unified memory. The active 32B parameters would stay hot in RAM while the other 968B page from SSD.

This is the trajectory: what costs $25/hr on cloud GPUs today will run on a desk for $0/hr as Apple Silicon RAM scales up.

---

## Speed Experiments (and what's next)

We tested three techniques to push past the current 30 tok/s ceiling. Here's what we found — and what someone with more RAM could unlock.

### What we tested on 16 GB

| Experiment | Result | Why |
|---|---|---|
| **Speculative decoding** (9B drafts, 35B verifies) | **GPU OOM** — both models need 15.9 GB, only 12.7 GB available | Needs 48 GB+ |
| **Multi-Token Prediction** (Qwen3.5 MTP) | No improvement (16.5 → 16.2 tok/s) | Homebrew llama.cpp doesn't compile MTP for Qwen3.5. Needs vLLM/SGLang |
| **SSD paging optimization** (mmap vs no-mmap) | ~3% improvement (28.6 → 29.4 tok/s) | macOS mmap already handles SSD prefetching well |
| **Thread tuning** (2/4/6/8 threads) | Diminishing returns past 4 threads | GPU-bound, not CPU-bound |

### What 48 GB+ unlocks

These techniques are **proven to work** but need more RAM than 16 GB:

**1. Speculative Decoding (est. 2-3x speedup)**

Use the 9B model as a fast "draft" model, the 35B verifies in batches. Needs both models in GPU memory simultaneously (~16 GB).

```bash
# Requires 48 GB+ Mac
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --model-draft ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 -t 4 \
    --draft-max 8 --draft-min 2
```

Expected: **60-90 tok/s** with 35B-class intelligence.

**2. Multi-Token Prediction (est. 1.5-2x speedup)**

Qwen3.5 was trained with MTP — it can predict 2-4 tokens per forward pass. Needs vLLM or SGLang instead of llama.cpp:

```bash
# Using vLLM (any GPU with enough VRAM)
vllm serve Qwen/Qwen3.5-35B-A3B --port 8000 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
```

Expected: **45-60 tok/s** on the same hardware.

**3. Expert Prefetching (the real "LLM in a Flash")**

Predict which MoE experts the next token will need and prefetch them from SSD while the GPU processes the current token. This completely hides SSD latency. Not yet implemented in any open-source framework — but the Apple paper describes the algorithm.

Expected: Paged inference matching in-RAM speed (~57 tok/s on the 35B).

### How to contribute

If you have a Mac with 48 GB+ RAM, you can test speculative decoding today:

1. Download both models (9B + 35B)
2. Run the speculative decoding command above
3. Report your tok/s vs baseline in an issue

If you have experience with vLLM on Apple Silicon, MTP support would be a major contribution.

---

## Benchmarks

212 math problems verified with SymPy (Qwen3.5-9B):

| Category | Score |
|---|---|
| Linear Algebra | **100%** (22/22) |
| Number Theory | **100%** (22/22) |
| Logic | **100%** (20/20) |
| Differential Equations | 95% |
| Geometry | 91% |
| Algebra | 86% |
| **Overall** | **86.3%** (183/212) |

---

## Common Issues

- **GPU OOM after long sessions**: Reboot your Mac to clear Metal GPU memory, then restart the server
- **Context overflow**: Clear sessions with `rm -rf ~/.picoclaw/workspace/sessions/`
- **Model swap hangs**: The server needs ~20s to load a new model — be patient
- **35B tool calling fails**: Expected — the IQ2_M quantization degrades instruction following. Auto-routing handles this by sending tool queries to the 9B

---

## License

MIT

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the models (Alibaba)
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** — inference engine (Georgi Gerganov)
- **[PicoClaw](https://github.com/sipeed/picoclaw)** — agent framework (Sipeed)
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[Rich](https://github.com/Textualize/rich)** — terminal UI (Will McGugan)
