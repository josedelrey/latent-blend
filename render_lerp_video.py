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

OUT_PREFIX_BASE = "videos/smooth_lerp"
COMFY_OUTPUT_DIR = os.path.expanduser("~/ComfyUI/output")
ENDPOINT_DIR = os.path.join(COMFY_OUTPUT_DIR, OUT_PREFIX_BASE)

WAIT_FOR_EACH = True
SAVE_SUFFIX = "_00001_"

# Phase lengths
STATIC_NUM = 120
TRANS_FRAMES = 60
MID_FRAMES = 120

# Denoise hyperparam
DENOISE_FAC = 0.55


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
# Node discovery / mutation
# ---------------------------------------------------------------------

def find_node_ids(prompt: ApiPrompt) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    blend_id: Optional[str] = None
    save_id: Optional[str] = None
    ksampler_id: Optional[str] = None

    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").lower()

        if save_id is None and ("saveimage" in ct or ct == "save image"):
            save_id = nid

        if blend_id is None and ("latent" in ct and "blend" in ct):
            blend_id = nid

        if ksampler_id is None and "ksampler" in ct:
            ksampler_id = nid

    return blend_id, save_id, ksampler_id


def set_blend_factor(prompt: ApiPrompt, blend_node_id: str, t: float) -> str:
    inputs = prompt[blend_node_id].setdefault("inputs", {})
    for key in ("blend_factor", "factor", "blend", "alpha", "t", "ratio", "mix", "strength"):
        if key in inputs and isinstance(inputs[key], (int, float)):
            inputs[key] = float(t)
            return key
    for key, val in inputs.items():
        if isinstance(val, (int, float)):
            inputs[key] = float(t)
            return key
    raise RuntimeError("No numeric blend parameter found on the Latent Blend node.")


def set_ksampler_denoise(prompt: ApiPrompt, ksampler_node_id: str, denoise: float) -> str:
    inputs = prompt[ksampler_node_id].setdefault("inputs", {})

    if "denoise" in inputs and isinstance(inputs["denoise"], (int, float)):
        inputs["denoise"] = float(denoise)
        return "denoise"

    for key in ("noise", "strength", "sigma"):
        if key in inputs and isinstance(inputs[key], (int, float)):
            inputs[key] = float(denoise)
            return key

    for key, val in inputs.items():
        if "denoise" in str(key).lower() and isinstance(val, (int, float)):
            inputs[key] = float(denoise)
            return key

    raise RuntimeError("Could not find a denoise-like numeric parameter on the KSampler node.")


def set_save_prefix(prompt: ApiPrompt, save_node_id: str, prefix: str) -> None:
    prompt[save_node_id].setdefault("inputs", {})["filename_prefix"] = prefix


# ---------------------------------------------------------------------
# Endpoint image resolution via ComfyUI API
# ---------------------------------------------------------------------

def find_load_image_nodes(prompt: ApiPrompt) -> List[str]:
    ids: List[str] = []
    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").lower()
        if "loadimage" in ct or ct == "load image":
            ids.append(nid)
    return ids


def get_load_image_filename(prompt: ApiPrompt, load_node_id: str) -> str:
    filename = prompt[load_node_id].get("inputs", {}).get("image")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError(f"LoadImage node {load_node_id} has no valid 'image' in inputs.")
    return filename


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

    blend_id, save_id, ksampler_id = find_node_ids(prompt)
    if blend_id is None or save_id is None or ksampler_id is None:
        print("Auto-detect failed. Available nodes (id -> class_type):")
        for nid, node in prompt.items():
            print(f"  {nid}: {node.get('class_type')}")
        raise RuntimeError(
            f"Auto-detect failed. blend_id={blend_id}, save_id={save_id}, ksampler_id={ksampler_id}."
        )

    load_ids = find_load_image_nodes(prompt)
    if len(load_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 LoadImage nodes, found {len(load_ids)}.")

    load_a_id, load_b_id = load_ids[0], load_ids[1]
    file_a = get_load_image_filename(prompt, load_a_id)
    file_b = get_load_image_filename(prompt, load_b_id)

    print(f"Blend node:    {blend_id}")
    print(f"KSampler node: {ksampler_id}")
    print(f"Save node:     {save_id}")
    print(f"Endpoint A:    {file_a}")
    print(f"Endpoint B:    {file_b}")
    print(
        f"Timeline: pre={STATIC_NUM}, head={TRANS_FRAMES}, mid={MID_FRAMES}, tail={TRANS_FRAMES}, post={STATIC_NUM}, "
        f"total={total_frames} | DENOISE_FAC={DENOISE_FAC}"
    )

    os.makedirs(ENDPOINT_DIR, exist_ok=True)

    t_start = time.time()

    a_bytes = download_input_image_bytes(file_a)
    b_bytes = download_input_image_bytes(file_b)

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
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        elapsed = time.time() - t_start
        print(f"[{frame_idx + 1:03d}/{total_frames}] head blend=0.0 denoise={denoise:.4f} prompt_id={prompt_id} (elapsed {mmss(elapsed)})")

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
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        elapsed = time.time() - t_start
        print(
            f"[{frame_idx + 1:03d}/{total_frames}] mid  blend={blend:.4f} denoise={DENOISE_FAC:.4f} prompt_id={prompt_id} (elapsed {mmss(elapsed)})"
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
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        elapsed = time.time() - t_start
        print(f"[{frame_idx + 1:03d}/{total_frames}] tail blend=1.0 denoise={denoise:.4f} prompt_id={prompt_id} (elapsed {mmss(elapsed)})")

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
