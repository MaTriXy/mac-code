#!/usr/bin/env python3
"""
Universal RAM Server for MoE Expert Sniping.

Runs on ANY machine (Mac, Linux, Windows, Raspberry Pi).
No MLX, no GPU, no Apple Silicon required.
Just RAM + Python + a network connection.

Loads expert weight files into RAM. Serves them over TCP.
The M4 Mac mini does all the compute.

Usage:
    python3 ram_server.py --expert-dir ~/models/experts/ --port 9000

Requirements:
    pip install numpy safetensors
    (that's it — no MLX, no torch, no GPU)
"""

import argparse
import json
import mmap
import os
import socket
import struct
import time
import numpy as np

# Protocol
MSG_FETCH = 0       # Request: give me these experts for this layer
MSG_WEIGHTS = 1     # Response: here are the weight bytes
MSG_PING = 2
MSG_PONG = 3
MSG_INFO = 4        # Request: what layers do you have?
MSG_INFO_RESP = 5
MSG_SHUTDOWN = 255

HEADER_SIZE = 16
HEADER_FMT = '<BBHI8x'  # type(1) + layer(1) + num_experts(2) + payload_size(4) + padding(8)


class RAMExpertStore:
    """
    Loads expert weight files into RAM as raw byte arrays.
    No framework needed — just numpy for memory management.
    """

    def __init__(self, expert_dir):
        self.expert_dir = expert_dir
        self.layers = {}  # layer_idx -> {key: numpy_array}
        self.layer_files = {}  # layer_idx -> mmap'd file

    def load_all(self):
        """Load all expert files into RAM."""
        print(f"Loading experts from {self.expert_dir}...")
        t0 = time.time()
        total_bytes = 0

        for fname in sorted(os.listdir(self.expert_dir)):
            if not fname.startswith("layer_"):
                continue

            layer_idx = int(fname.split("_")[1].split(".")[0])
            path = os.path.join(self.expert_dir, fname)
            size = os.path.getsize(path)
            total_bytes += size

            if fname.endswith(".safetensors"):
                self._load_safetensors(layer_idx, path)
            elif fname.endswith(".bin"):
                self._load_binary(layer_idx, path)

            print(f"  Layer {layer_idx}: {size/1e6:.1f} MB")

        elapsed = time.time() - t0
        print(f"  Total: {len(self.layers)} layers, {total_bytes/1e9:.2f} GB in {elapsed:.1f}s")

    def _load_safetensors(self, layer_idx, path):
        """Load a safetensors file into RAM as numpy arrays."""
        from safetensors import safe_open
        data = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                data[key] = f.get_tensor(key)
        self.layers[layer_idx] = data

    def _load_binary(self, layer_idx, path):
        """Memory-map a binary expert file."""
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.layer_files[layer_idx] = (f, mm)
        # Parse header to get layout
        header_size = 16384  # 16KB header
        header_raw = mm[:header_size]
        header = json.loads(header_raw.rstrip(b'\x00'))
        self.layers[layer_idx] = {"_header": header, "_mmap": mm}

    def get_expert_bytes(self, layer_idx, expert_ids, projections=None):
        """
        Get raw bytes for specific experts from a layer.
        Returns a dict of {key: bytes} for the requested experts.
        """
        if projections is None:
            projections = ["gate_proj", "up_proj", "down_proj"]

        data = self.layers.get(layer_idx)
        if data is None:
            return {}

        result = {}
        for proj in projections:
            for comp in ["weight", "scales", "biases"]:
                key = f"{proj}.{comp}"
                if key in data:
                    full_array = data[key]  # [256, out, in]
                    # Extract only requested experts
                    selected = full_array[expert_ids]  # [num_experts, out, in]
                    result[key] = selected.tobytes()
                    if not result.get("_shapes"):
                        result["_shapes"] = {}
                    result["_shapes"][key] = {
                        "shape": list(selected.shape),
                        "dtype": str(selected.dtype),
                    }

        return result

    def get_layer_info(self, layer_idx):
        """Get metadata about a layer's experts."""
        data = self.layers.get(layer_idx)
        if data is None:
            return None
        info = {}
        for key, arr in data.items():
            if key.startswith("_"):
                continue
            info[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        return info


def handle_client(conn, addr, store):
    """Handle a single client connection."""
    print(f"Client connected: {addr}")

    try:
        while True:
            # Read header
            header_data = b''
            while len(header_data) < HEADER_SIZE:
                chunk = conn.recv(HEADER_SIZE - len(header_data))
                if not chunk:
                    raise ConnectionError("Closed")
                header_data += chunk

            msg_type, layer_idx, num_experts, payload_size = struct.unpack(HEADER_FMT, header_data)

            if msg_type == MSG_PING:
                resp = struct.pack(HEADER_FMT, MSG_PONG, 0, 0, 0)
                conn.sendall(resp)
                continue

            if msg_type == MSG_INFO:
                info = {
                    "layers": sorted(store.layers.keys()),
                    "num_layers": len(store.layers),
                }
                if store.layers:
                    first = next(iter(store.layers.values()))
                    sample_key = next(k for k in first if not k.startswith("_"))
                    info["sample_shape"] = list(first[sample_key].shape)
                info_bytes = json.dumps(info).encode()
                resp = struct.pack(HEADER_FMT, MSG_INFO_RESP, 0, 0, len(info_bytes))
                conn.sendall(resp + info_bytes)
                continue

            if msg_type == MSG_SHUTDOWN:
                print("Shutdown requested")
                return False

            if msg_type == MSG_FETCH:
                # Read expert IDs from payload
                payload = b''
                while len(payload) < payload_size:
                    chunk = conn.recv(min(65536, payload_size - len(payload)))
                    if not chunk:
                        raise ConnectionError("Closed during payload")
                    payload += chunk

                expert_ids = list(struct.unpack(f'<{num_experts}I', payload[:num_experts * 4]))

                t0 = time.time()

                # Get expert weight bytes
                expert_data = store.get_expert_bytes(layer_idx, expert_ids)

                # Serialize response: shapes_json + raw bytes
                shapes_json = json.dumps(expert_data.get("_shapes", {})).encode()
                raw_bytes = b''
                for key in sorted(k for k in expert_data if not k.startswith("_")):
                    raw_bytes += expert_data[key]

                # Pack: shapes_len(4) + shapes_json + raw_bytes
                response_payload = struct.pack('<I', len(shapes_json)) + shapes_json + raw_bytes

                resp_header = struct.pack(HEADER_FMT, MSG_WEIGHTS, layer_idx, num_experts, len(response_payload))
                conn.sendall(resp_header + response_payload)

                elapsed = (time.time() - t0) * 1000
                total_mb = len(raw_bytes) / 1e6
                if elapsed > 0:
                    throughput = total_mb / (elapsed / 1000)
                else:
                    throughput = 0
                # Only log occasionally to avoid spam
                if layer_idx == 0:
                    print(f"  L{layer_idx}: {num_experts} experts, {total_mb:.1f} MB, {elapsed:.1f}ms ({throughput:.0f} MB/s)")

    except (ConnectionError, BrokenPipeError) as e:
        print(f"Client disconnected: {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Universal RAM Server for MoE Expert Sniping")
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    store = RAMExpertStore(args.expert_dir)
    store.load_all()

    # Print system info
    import platform
    print(f"\n{'='*50}")
    print(f"  RAM Server Ready")
    print(f"  Machine: {platform.node()}")
    print(f"  OS: {platform.system()} {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Layers: {sorted(store.layers.keys())}")
    print(f"  Listening: {args.host}:{args.port}")
    print(f"{'='*50}\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Large send/recv buffers for throughput
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((args.host, args.port))
    sock.listen(2)

    while True:
        conn, addr = sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        keep_running = handle_client(conn, addr, store)
        conn.close()
        if not keep_running:
            break


if __name__ == "__main__":
    main()
