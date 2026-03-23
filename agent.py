#!/usr/bin/env python3
"""
pico-mini — Local AI agent on a Mac mini
"""

import json, sys, os, time, subprocess, re, threading, queue
import urllib.request, random

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding
from rich.columns import Columns

SERVER = os.environ.get("LLAMA_URL", "http://localhost:8000")
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
console = Console()

# ── ANSI strip ─────────────────────────────────────
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\r')
def strip_ansi(text):
    return ANSI_RE.sub('', text)

# ── activity animations ───────────────────────────
FRAMES_THINK = ["◐", "◓", "◑", "◒"]
FRAMES_SEARCH = ["◜ ", " ◝", " ◞", "◟ "]
FRAMES_FETCH = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
FRAMES_EXEC = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
FRAMES_DONE = ["✓"]

PHASE_CONFIG = {
    "thinking":           ("bright_cyan",   FRAMES_THINK,  "brain is cooking"),
    "reading message":    ("bright_cyan",   FRAMES_THINK,  "parsing your request"),
    "searching the web":  ("bright_green",  FRAMES_SEARCH, "querying DuckDuckGo"),
    "fetching page":      ("bright_yellow", FRAMES_FETCH,  "downloading content"),
    "running command":    ("bright_red",    FRAMES_EXEC,   "executing shell"),
    "reading file":       ("bright_yellow", FRAMES_FETCH,  "reading from disk"),
    "writing file":       ("bright_yellow", FRAMES_FETCH,  "writing to disk"),
    "compressing context":("dim",           FRAMES_THINK,  "shrinking history"),
    "finishing up":       ("bright_green",  FRAMES_DONE,   "almost done"),
}

# ── live working display ──────────────────────────
class WorkingDisplay:
    def __init__(self):
        self.events = []       # (timestamp, phase, detail) — full timeline
        self.phase = "thinking"
        self.frame = 0
        self.start_time = time.time()

    def add_log(self, line):
        clean = strip_ansi(line).strip()
        if not clean:
            return

        lower = clean.lower()
        new_phase = None
        detail = ""

        # Detect phase transitions from picoclaw logs
        if "processing message" in lower:
            new_phase = "reading message"
            detail = "received your input"
        elif "llm_request" in lower:
            new_phase = "thinking"
            # Try to extract model name
            if "model=" in lower:
                m = re.search(r'model=(\S+)', clean)
                detail = f"model={m.group(1)}" if m else "calling LLM"
            else:
                detail = "generating response"
        elif "tool_call" in lower:
            # Try to extract tool name
            if "web_search" in lower or "duckduckgo" in lower:
                new_phase = "searching the web"
                detail = "DuckDuckGo search"
            elif "web_fetch" in lower or "fetch" in lower:
                new_phase = "fetching page"
                m = re.search(r'url[=:]"?([^"\s,}]+)', clean, re.IGNORECASE)
                detail = m.group(1)[:60] if m else "downloading"
            elif "exec" in lower:
                new_phase = "running command"
                detail = "shell execution"
            elif "read_file" in lower:
                new_phase = "reading file"
                detail = ""
            elif "write_file" in lower:
                new_phase = "writing file"
                detail = ""
            else:
                new_phase = "thinking"
                detail = "tool call"
        elif "tool_result" in lower:
            detail = "got result"
        elif "context_compress" in lower:
            new_phase = "compressing context"
            detail = "trimming conversation history"
        elif "turn_end" in lower:
            new_phase = "finishing up"
            detail = "wrapping up"

        if new_phase:
            self.phase = new_phase
            self.events.append((time.time() - self.start_time, new_phase, detail))
        elif detail:
            self.events.append((time.time() - self.start_time, self.phase, detail))

    def render(self):
        self.frame += 1
        elapsed = time.time() - self.start_time
        cfg = PHASE_CONFIG.get(self.phase, ("bright_cyan", FRAMES_THINK, ""))
        color, frames, hint = cfg
        spinner = frames[self.frame % len(frames)]

        out = Text()

        # Main status line with spinner
        out.append(f"  {spinner} ", style=f"bold {color}")
        out.append(self.phase, style=f"bold {color}")
        out.append(f"  {elapsed:.0f}s", style="dim")
        if hint:
            out.append(f"  {hint}", style="dim italic")
        out.append("\n")

        # Show recent events as a live timeline
        recent = self.events[-5:]
        for ts, phase, detail in recent:
            icon = "│"
            # Color-code by phase
            pcfg = PHASE_CONFIG.get(phase, ("dim", [], ""))
            pcolor = pcfg[0]

            out.append(f"  {icon} ", style="dim")
            out.append(f"{ts:5.1f}s ", style="dim")

            # Phase badge
            badge = phase[:15].ljust(15)
            out.append(badge, style=f"{pcolor}")

            if detail:
                out.append(f"  {detail[:50]}", style="dim italic")
            out.append("\n")

        # Bottom connector
        if recent:
            out.append("  ╰─", style="dim")
            dots = "─" * min(int(elapsed * 2) % 20, 20)
            out.append(dots, style="dim")
            pulse = "●" if self.frame % 4 < 2 else "○"
            out.append(f" {pulse}", style=f"bold {color}")
            out.append("\n")

        return out

# ── detect model ───────────────────────────────────
def detect_model():
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "Qwen3.5-35B-A3B", "MoE 34.7B · 3B active · IQ2_M"
        elif "9B" in alias:
            return "Qwen3.5-9B", "8.95B dense · Q4_K_M"
        return alias.replace(".gguf", "").split("/")[-1], "local"
    except Exception:
        return "offline", ""

# ── streaming chat (raw mode) ─────────────────────
def stream_llm(messages):
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    start = time.time()
    tokens = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        while True:
            ch = resp.read(1)
            if not ch:
                break
            buf += ch.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    return full, tokens, time.time() - start
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full += c
                        tokens += 1
                        yield c
                except Exception:
                    pass

    return full, tokens, time.time() - start

# ── picoclaw agent call with LIVE log streaming ───
def picoclaw_call_live(message, session="pico-mini"):
    """Run picoclaw with real-time log streaming into animated display."""
    # Use -d (debug) flag so picoclaw emits detailed logs to stdout
    cmd = [PICOCLAW, "agent", "-m", message, "-s", session, "-d"]
    display = WorkingDisplay()
    all_lines = []

    # Launch with Popen — picoclaw writes everything to stdout
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    # Read stdout line-by-line in a thread for real-time updates
    def read_output():
        try:
            for line in proc.stdout:
                all_lines.append(line)
                display.add_log(line)
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    # Animate while process runs
    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
        while proc.poll() is None:
            live.update(display.render())
            time.sleep(0.12)
        # Give reader a moment to finish
        time.sleep(0.3)
        live.update(display.render())

    reader.join(timeout=2)

    # Parse: join all output, strip ANSI, take LAST lobster section (the actual response)
    raw = "".join(all_lines)
    clean = strip_ansi(raw)
    parts = clean.split("\U0001f99e")
    if len(parts) >= 2:
        # Last lobster section is the actual AI response
        response = parts[-1].strip()
    else:
        # No lobster found — fallback: skip log/banner lines
        lines = clean.split("\n")
        resp = []
        for line in lines:
            s = line.strip()
            if s and not any(k in s for k in [
                "██", "╔", "╚", "╝", "║", "DBG", "INF", "ERR", "WRN",
                "pkg/", "cmd/", "Debug mode", "picoclaw",
            ]):
                resp.append(s)
        response = "\n".join(resp[-20:]).strip()  # take last 20 clean lines

    return response, display.events

# ── banner ─────────────────────────────────────────
def print_banner(model_name, model_detail):
    console.print()
    logo = Text()
    logo.append("  \U0001f34e ", style="default")
    logo.append("pico", style="bold bright_cyan")
    logo.append("-", style="dim")
    logo.append("mini", style="bold bright_yellow")
    console.print(logo)

    sub = Text()
    sub.append("  AI agent running locally on your Mac", style="dim italic")
    console.print(sub)
    console.print()

    rows = [
        ("model", model_name, model_detail),
        ("tools", "search · fetch · exec · files", ""),
        ("cost", "$0.00/hr", "Apple M4 Metal · localhost:8000"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:6s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)

    console.print()
    console.print(Rule(style="dim"))
    console.print()

# ── render helpers ─────────────────────────────────
def render_speed(tokens, elapsed):
    if elapsed <= 0 or tokens <= 0:
        return
    speed = tokens / elapsed
    clr = "bright_green" if speed > 20 else "yellow" if speed > 10 else "red"
    s = Text()
    s.append(f"  {speed:.1f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
    console.print(s)

def render_timeline(events):
    """Show a compact summary of what the agent did."""
    if not events:
        return
    # Deduplicate consecutive same-phase entries
    summary = []
    last_phase = None
    for ts, phase, detail in events:
        if phase != last_phase:
            summary.append((ts, phase, detail))
            last_phase = phase

    if len(summary) <= 1:
        return

    t = Text()
    t.append("  ", style="dim")
    for i, (ts, phase, detail) in enumerate(summary):
        cfg = PHASE_CONFIG.get(phase, ("dim", [], ""))
        color = cfg[0]
        label = phase.split()[-1]  # last word
        t.append(label, style=f"{color}")
        if i < len(summary) - 1:
            t.append(" → ", style="dim")
    console.print(t)

# ── commands ───────────────────────────────────────
def print_help():
    cmds = [
        ("/agent", "Agent mode — tools enabled (default)"),
        ("/raw", "Raw mode — direct streaming, no tools"),
        ("/clear", "Clear conversation"),
        ("/stats", "Session statistics"),
        ("/model", "Current model info"),
        ("/tools", "List available tools"),
        ("/system <msg>", "Set system prompt"),
        ("/quit", "Exit"),
    ]
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold bright_cyan", width=16)
    t.add_column(style="dim")
    for cmd, desc in cmds:
        t.add_row(cmd, desc)
    console.print(t)
    console.print()

# ── main ───────────────────────────────────────────
def main():
    model_name, model_detail = detect_model()
    console.clear()
    print_banner(model_name, model_detail)

    messages = []
    session_tokens = 0
    session_time = 0.0
    session_turns = 0
    session_id = f"pm-{int(time.time())}"
    use_agent = True

    while True:
        try:
            tag = "agent" if use_agent else "raw"
            console.print(f"  [dim]{tag}[/] [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/clear":
            messages.clear()
            session_id = f"pm-{int(time.time())}"
            console.clear()
            print_banner(model_name, model_detail)
            console.print("  [dim]cleared.[/]\n")
            continue
        elif cmd == "/stats":
            avg = session_tokens / session_time if session_time > 0 else 0
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_column(style="bold bright_cyan", width=12)
            t.add_column()
            t.add_row("turns", str(session_turns))
            t.add_row("tokens", f"{session_tokens:,}")
            t.add_row("time", f"{session_time:.1f}s")
            t.add_row("avg speed", f"{avg:.1f} tok/s")
            t.add_row("mode", tag)
            console.print(t)
            console.print()
            continue
        elif cmd == "/model":
            model_name, model_detail = detect_model()
            console.print(f"  [bold white]{model_name}[/]  [dim]{model_detail}[/]\n")
            continue
        elif cmd == "/tools":
            for name, desc in [
                ("web_search", "DuckDuckGo"), ("web_fetch", "read URLs"),
                ("exec", "shell commands"), ("read_file", "local files"),
                ("write_file", "create files"), ("edit_file", "modify files"),
                ("list_dir", "browse dirs"), ("subagent", "spawn tasks"),
            ]:
                t = Text()
                t.append("  ▸ ", style="bright_cyan")
                t.append(name, style="bold bright_cyan")
                t.append(f"  {desc}", style="dim")
                console.print(t)
            console.print()
            continue
        elif cmd == "/agent":
            use_agent = True
            console.print("  [dim]agent mode (tools enabled)[/]\n")
            continue
        elif cmd == "/raw":
            use_agent = False
            console.print("  [dim]raw mode (streaming, no tools)[/]\n")
            continue
        elif cmd in ("/help", "/?"):
            print_help()
            continue
        elif cmd.startswith("/system "):
            sys_msg = user_input[8:].strip()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = sys_msg
            else:
                messages.insert(0, {"role": "system", "content": sys_msg})
            console.print(f"  [dim italic]system: {sys_msg[:80]}[/]\n")
            continue

        console.print()

        # ── agent mode ─────────────────────────────
        if use_agent:
            start = time.time()
            response, events = picoclaw_call_live(user_input, session=session_id)
            elapsed = time.time() - start

            if response:
                # Show what the agent did
                render_timeline(events)
                console.print()

                # Render response
                if any(c in response for c in ["##", "**", "```", "- ", "1. ", "* "]):
                    console.print(Padding(Markdown(response), (0, 2)))
                else:
                    for line in response.split("\n"):
                        console.print(f"  {line}")
                console.print()
                tokens_est = len(response.split())
                render_speed(tokens_est, elapsed)
                session_tokens += tokens_est
                session_time += elapsed
                session_turns += 1
            else:
                console.print("  [bold red]no response[/]")

        # ── raw streaming mode ─────────────────────
        else:
            messages.append({"role": "user", "content": user_input})
            full = ""
            tokens = 0
            start = time.time()

            try:
                display = WorkingDisplay()
                display.phase = "thinking"
                first_token = True

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
                    gen = stream_llm(messages)
                    for chunk in gen:
                        if isinstance(chunk, str):
                            if first_token:
                                first_token = False
                                live.stop()
                                console.print("  ", end="")
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1

                elapsed = time.time() - start
                if any(c in full for c in ["##", "**", "```", "- ", "1. "]):
                    console.print("\n")
                    console.print(Padding(Markdown(full), (0, 2)))
                else:
                    console.print("\n")
                render_speed(tokens, elapsed)
                session_tokens += tokens
                session_time += elapsed
                session_turns += 1
                messages.append({"role": "assistant", "content": full})

            except Exception as e:
                console.print(f"  [bold red]{e}[/]")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

        console.print()

    # ── exit ───────────────────────────────────────
    console.print()
    if session_turns > 0:
        avg = session_tokens / session_time if session_time > 0 else 0
        console.print(
            f"  \U0001f34e [bold bright_cyan]pico[/][dim]-[/][bold bright_yellow]mini[/]"
            f"  [dim]{session_turns} turns · {session_tokens:,} tokens · {avg:.1f} tok/s[/]"
        )
    console.print()

if __name__ == "__main__":
    main()
