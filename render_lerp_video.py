"""
ComfyUI latent morph frame renderer (API-driven).

Timeline layout (5 phases)
0) Pre   (STATIC_NUM):    static A (exact input bytes)
1) Head  (TRANS_FRAMES):  blend=0.0, denoise ramps   0.0 -> DENOISE_FAC (LINEAR)
2) Mid   (MID_FRAMES):    blend sweeps 0.0 -> 1.0, denoise fixed at DENOISE_FAC
3) Tail  (TRANS_FRAMES):  blend=1.0, denoise ramps   DENOISE_FAC -> 0.0 (LINEAR)
4) Post  (STATIC_NUM):    static B (exact input bytes)

Naming
- Frames are numbered globally: frame_00000{SAVE_SUFFIX}.png ... frame_<TOTAL-1>{SAVE_SUFFIX}.png
- Pre/Post frames are written as exact bytes (no ComfyUI render).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "cursed1.json"

OUT_PREFIX_BASE = "videos/cursedmidjourney_lerp_high"
COMFY_OUTPUT_DIR = os.path.expanduser("~/ComfyUI/output")
ENDPOINT_DIR = os.path.join(COMFY_OUTPUT_DIR, OUT_PREFIX_BASE)

WAIT_FOR_EACH = True
SAVE_SUFFIX = "_00001_"

# Phase lengths
STATIC_NUM = 60
TRANS_FRAMES = 40
MID_FRAMES = 80

# Denoise hyperparam
DENOISE_FAC = 0.65

# KSampler CFG
KSAMPLER_CFG = 3.5

# LoRA strengths
LORA_STRENGTH_MODEL = 0.9
LORA_STRENGTH_CLIP = 0.9

# Pick filenames that exist under ~/ComfyUI/input/
INPUT_A_FILENAME = "37.png"
INPUT_B_FILENAME = "7.png"

# ---------------------------------------------------------------------
# Confirmed key names (from your discovery output)
# ---------------------------------------------------------------------

LOAD_IMAGE_KEY = "image"
SAVE_PREFIX_KEY = "filename_prefix"
BLEND_KEY = "blend_factor"
DENOISE_KEY = "denoise"
CFG_KEY = "cfg"
LORA_MODEL_KEY = "strength_model"
LORA_CLIP_KEY = "strength_clip"

# ---------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------
def mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------
# Workflow loading / normalization
# ---------------------------------------------------------------------
ApiPrompt = Dict[str, Dict[str, Any]]


def load_as_api_prompt(path: str) -> ApiPrompt:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        node_like = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v and "inputs" in v
        }
        if node_like:
            return node_like

    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        prompt: ApiPrompt = {}
        for n in data["nodes"]:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            class_type = n.get("type") or n.get("class_type")
            inputs = n.get("inputs", {})
            if nid is None or not class_type:
                continue
            prompt[str(nid)] = {"class_type": class_type, "inputs": inputs}

        if prompt and any(
            isinstance(v.get("inputs"), dict) and v["inputs"]
            for v in prompt.values()
        ):
            return prompt

        raise RuntimeError(
            "Workflow JSON doesn't contain sufficient API 'inputs'. "
            "Export/copy an API prompt JSON (node_id -> {class_type, inputs})."
        )

    raise RuntimeError("Unrecognized workflow JSON format.")


# ---------------------------------------------------------------------
# Node resolution by class_type + required keys
# ---------------------------------------------------------------------

def _ct(node: Dict[str, Any]) -> str:
    return (node.get("class_type") or "").lower()


def _inputs(node: Dict[str, Any]) -> Dict[str, Any]:
    ins = node.get("inputs", {})
    return ins if isinstance(ins, dict) else {}


def find_single_node(
    prompt: ApiPrompt,
    *,
    class_type_contains: Optional[str] = None,
    required_input_keys: Optional[List[str]] = None,
    forbidden_class_type_contains: Optional[str] = None,
    label: str = "node",
) -> str:
    required_input_keys = required_input_keys or []
    matches: List[str] = []

    for nid, node in prompt.items():
        ct = _ct(node)
        ins = _inputs(node)

        if class_type_contains is not None and class_type_contains not in ct:
            continue
        if forbidden_class_type_contains is not None and forbidden_class_type_contains in ct:
            continue
        if any(k not in ins for k in required_input_keys):
            continue

        matches.append(nid)

    if not matches:
        # Print a compact inventory for debugging.
        print(f"[ERROR] Could not resolve {label}. Criteria:")
        print(f"  class_type_contains={class_type_contains!r}")
        print(f"  required_input_keys={required_input_keys}")
        print("  Available nodes (id -> class_type -> input_keys):")
        for nid, node in prompt.items():
            ins = _inputs(node)
            print(f"    {nid}: {node.get('class_type')} -> {sorted(list(ins.keys()))}")
        raise RuntimeError(f"Could not resolve {label} (no matches).")

    if len(matches) > 1:
        print(f"[ERROR] Ambiguous {label}: multiple matches {matches}")
        for nid in matches:
            node = prompt[nid]
            ins = _inputs(node)
            print(f"  {nid}: {node.get('class_type')} -> {sorted(list(ins.keys()))}")
        raise RuntimeError(f"Could not resolve {label} (ambiguous: {len(matches)} matches).")

    return matches[0]


def find_two_loadimage_nodes(prompt: ApiPrompt) -> Tuple[str, str]:
    # We want exactly two LoadImage nodes that accept the 'image' key.
    matches: List[str] = []
    for nid, node in prompt.items():
        ct = _ct(node)
        ins = _inputs(node)
        if ("loadimage" in ct or ct == "load image") and (LOAD_IMAGE_KEY in ins):
            matches.append(nid)

    if len(matches) != 2:
        print(f"[ERROR] Expected exactly 2 LoadImage nodes with key '{LOAD_IMAGE_KEY}', found {len(matches)}: {matches}")
        for nid in matches:
            node = prompt[nid]
            print(f"  {nid}: {node.get('class_type')} -> {sorted(list(_inputs(node).keys()))}")
        print("  Available nodes (id -> class_type -> input_keys):")
        for nid, node in prompt.items():
            print(f"    {nid}: {node.get('class_type')} -> {sorted(list(_inputs(node).keys()))}")
        raise RuntimeError("Could not resolve the two LoadImage nodes.")
    return matches[0], matches[1]


def resolve_graph_ids(prompt: ApiPrompt) -> Tuple[str, str, str, str, str, str]:
    """
    Returns:
      lora_id, load_a_id, load_b_id, ksampler_id, save_id, blend_id
    """
    lora_id = find_single_node(
        prompt,
        class_type_contains="lora",
        required_input_keys=[LORA_MODEL_KEY, LORA_CLIP_KEY],
        label="LoRA loader node",
    )

    load_a_id, load_b_id = find_two_loadimage_nodes(prompt)

    ksampler_id = find_single_node(
        prompt,
        class_type_contains="ksampler",
        required_input_keys=[DENOISE_KEY, CFG_KEY],
        label="KSampler node",
    )

    save_id = find_single_node(
        prompt,
        class_type_contains="saveimage",
        required_input_keys=[SAVE_PREFIX_KEY],
        label="SaveImage node",
    )

    blend_id = find_single_node(
        prompt,
        class_type_contains="blend",
        required_input_keys=[BLEND_KEY],
        label="Latent Blend node",
    )

    return lora_id, load_a_id, load_b_id, ksampler_id, save_id, blend_id


# ---------------------------------------------------------------------
# Direct setters (fixed key names)
# ---------------------------------------------------------------------

def require_node(prompt: ApiPrompt, node_id: str) -> Dict[str, Any]:
    if node_id not in prompt:
        raise KeyError(f"Node id {node_id} not found in prompt.")
    node = prompt[node_id]
    if not isinstance(node, dict):
        raise TypeError(f"Node id {node_id} is not a dict.")
    node.setdefault("inputs", {})
    if not isinstance(node["inputs"], dict):
        raise TypeError(f"Node id {node_id} inputs is not a dict.")
    return node


def set_load_image_filename(prompt: ApiPrompt, load_node_id: str, filename: str) -> None:
    node = require_node(prompt, load_node_id)
    node["inputs"][LOAD_IMAGE_KEY] = filename


def set_save_prefix(prompt: ApiPrompt, save_node_id: str, prefix: str) -> None:
    node = require_node(prompt, save_node_id)
    node["inputs"][SAVE_PREFIX_KEY] = prefix


def set_blend_factor(prompt: ApiPrompt, blend_node_id: str, t: float) -> None:
    node = require_node(prompt, blend_node_id)
    node["inputs"][BLEND_KEY] = float(t)


def set_ksampler_denoise(prompt: ApiPrompt, ksampler_node_id: str, denoise: float) -> None:
    node = require_node(prompt, ksampler_node_id)
    node["inputs"][DENOISE_KEY] = float(denoise)


def set_ksampler_cfg(prompt: ApiPrompt, ksampler_node_id: str, cfg: float) -> None:
    node = require_node(prompt, ksampler_node_id)
    node["inputs"][CFG_KEY] = float(cfg)


def set_lora_strengths(prompt: ApiPrompt, lora_node_id: str, strength_model: float, strength_clip: float) -> None:
    node = require_node(prompt, lora_node_id)
    node["inputs"][LORA_MODEL_KEY] = float(strength_model)
    node["inputs"][LORA_CLIP_KEY] = float(strength_clip)


# ---------------------------------------------------------------------
# Endpoint image resolution via ComfyUI API
# ---------------------------------------------------------------------

def download_input_image_bytes(filename: str) -> bytes:
    r = requests.get(
        f"{COMFY_URL}/view",
        params={"filename": filename, "type": "input"},
        timeout=60,
    )
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------
# ComfyUI API helpers
# ---------------------------------------------------------------------

def wait_for_prompt_done(prompt_id: str, poll_s: float = 0.5) -> Dict[str, Any]:
    while True:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data and data[prompt_id].get("outputs"):
            return data[prompt_id]
        time.sleep(poll_s)


def submit_prompt(job: ApiPrompt) -> str:
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": job}, timeout=60)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def write_static_frames(img_bytes: bytes, start_idx: int, count: int) -> None:
    for i in range(count):
        frame_idx = start_idx + i
        path = os.path.join(ENDPOINT_DIR, f"frame_{frame_idx:05d}{SAVE_SUFFIX}.png")
        with open(path, "wb") as f:
            f.write(img_bytes)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if STATIC_NUM < 0 or TRANS_FRAMES < 0 or MID_FRAMES < 0:
        raise ValueError("STATIC_NUM, TRANS_FRAMES and MID_FRAMES must be >= 0.")
    if MID_FRAMES < 2:
        raise ValueError("MID_FRAMES must be >= 2 to sweep blend factor 0->1.")
    if TRANS_FRAMES == 1:
        raise ValueError("TRANS_FRAMES must be 0 or >= 2 for a denoise ramp.")
    if not (0.0 <= DENOISE_FAC <= 1.0):
        raise ValueError("DENOISE_FAC must be in [0, 1].")

    total_frames = 2 * STATIC_NUM + 2 * TRANS_FRAMES + MID_FRAMES
    prompt = load_as_api_prompt(WORKFLOW_PATH)

    lora_id, load_a_id, load_b_id, ksampler_id, save_id, blend_id = resolve_graph_ids(prompt)

    # Override the workflow endpoints
    set_load_image_filename(prompt, load_a_id, INPUT_A_FILENAME)
    set_load_image_filename(prompt, load_b_id, INPUT_B_FILENAME)

    # Set LoRA strengths once on base prompt (inherited by per-frame clones)
    set_lora_strengths(prompt, lora_id, LORA_STRENGTH_MODEL, LORA_STRENGTH_CLIP)

    print("[RESOLVED IDS]")
    print(f"  LoRA loader: {lora_id} (expects keys: {LORA_MODEL_KEY}, {LORA_CLIP_KEY})")
    print(f"  LoadImage A: {load_a_id} (key: {LOAD_IMAGE_KEY})")
    print(f"  LoadImage B: {load_b_id} (key: {LOAD_IMAGE_KEY})")
    print(f"  KSampler:    {ksampler_id} (keys: {DENOISE_KEY}, {CFG_KEY})")
    print(f"  SaveImage:   {save_id} (key: {SAVE_PREFIX_KEY})")
    print(f"  Blend:       {blend_id} (key: {BLEND_KEY})")
    print(
        f"Timeline: pre={STATIC_NUM}, head={TRANS_FRAMES}, mid={MID_FRAMES}, tail={TRANS_FRAMES}, post={STATIC_NUM}, "
        f"total={total_frames} | DENOISE_FAC={DENOISE_FAC} | CFG={KSAMPLER_CFG} | "
        f"LoRA(model={LORA_STRENGTH_MODEL}, clip={LORA_STRENGTH_CLIP})"
    )

    os.makedirs(ENDPOINT_DIR, exist_ok=True)

    t_start = time.time()

    a_bytes = download_input_image_bytes(INPUT_A_FILENAME)
    b_bytes = download_input_image_bytes(INPUT_B_FILENAME)

    # Phase 0: Pre (static A)
    write_static_frames(a_bytes, start_idx=0, count=STATIC_NUM)

    # Phase 1: Head — blend=0.0, denoise ramps 0.0 -> DENOISE_FAC (LINEAR)
    head_start = STATIC_NUM
    for i in range(TRANS_FRAMES):
        frame_idx = head_start + i
        if TRANS_FRAMES == 0:
            break

        u = 0.0 if TRANS_FRAMES == 1 else (i / (TRANS_FRAMES - 1))
        denoise = lerp(0.0, DENOISE_FAC, u)

        job: ApiPrompt = json.loads(json.dumps(prompt))
        set_blend_factor(job, blend_id, 0.0)
        set_ksampler_denoise(job, ksampler_id, denoise)
        set_ksampler_cfg(job, ksampler_id, KSAMPLER_CFG)
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        elapsed = time.time() - t_start
        print(
            f"[{frame_idx + 1:03d}/{total_frames}] head blend=0.0 denoise={denoise:.4f} "
            f"prompt_id={prompt_id} (elapsed {mmss(elapsed)})"
        )

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    # Phase 2: Mid — blend sweeps 0.0 -> 1.0, denoise fixed
    mid_start = STATIC_NUM + TRANS_FRAMES
    for j in range(MID_FRAMES):
        frame_idx = mid_start + j
        blend = j / (MID_FRAMES - 1)

        job: ApiPrompt = json.loads(json.dumps(prompt))
        set_blend_factor(job, blend_id, blend)
        set_ksampler_denoise(job, ksampler_id, DENOISE_FAC)
        set_ksampler_cfg(job, ksampler_id, KSAMPLER_CFG)
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        elapsed = time.time() - t_start
        print(
            f"[{frame_idx + 1:03d}/{total_frames}] mid  blend={blend:.4f} denoise={DENOISE_FAC:.4f} "
            f"prompt_id={prompt_id} (elapsed {mmss(elapsed)})"
        )

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    # Phase 3: Tail — blend=1.0, denoise ramps DENOISE_FAC -> 0.0 (LINEAR)
    tail_start = STATIC_NUM + TRANS_FRAMES + MID_FRAMES
    for i in range(TRANS_FRAMES):
        frame_idx = tail_start + i

        u = 0.0 if TRANS_FRAMES == 1 else (i / (TRANS_FRAMES - 1))
        denoise = lerp(DENOISE_FAC, 0.0, u)

        job: ApiPrompt = json.loads(json.dumps(prompt))
        set_blend_factor(job, blend_id, 1.0)
        set_ksampler_denoise(job, ksampler_id, denoise)
        set_ksampler_cfg(job, ksampler_id, KSAMPLER_CFG)
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        elapsed = time.time() - t_start
        print(
            f"[{frame_idx + 1:03d}/{total_frames}] tail blend=1.0 denoise={denoise:.4f} "
            f"prompt_id={prompt_id} (elapsed {mmss(elapsed)})"
        )

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    # Phase 4: Post (static B)
    post_start = STATIC_NUM + TRANS_FRAMES + MID_FRAMES + TRANS_FRAMES
    write_static_frames(b_bytes, start_idx=post_start, count=STATIC_NUM)

    total = time.time() - t_start
    print(f"[TIME] total blend time {mmss(total)}")
    print("Done: all frames generated.")


if __name__ == "__main__":
    main()
