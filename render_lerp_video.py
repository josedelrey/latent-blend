"""
ComfyUI latent morph frame renderer (API-driven).

Timeline layout (3 phases)
1) Head (TRANS_FRAMES): render TRANS_FRAMES frames with blend factor fixed at 0.0
2) Middle (MID_FRAMES): render MID_FRAMES frames sweeping blend factor 0.0 -> 1.0
3) Tail (TRANS_FRAMES): render TRANS_FRAMES frames with blend factor fixed at 1.0

Notes
- Total frames = 2 * TRANS_FRAMES + MID_FRAMES
- Filenames are frame_00000.png ... frame_<TOTAL-1>.png (zero-padded)
- Endpoint frames are also written as exact input bytes:
  - frame_00000.png (A)
  - frame_<TOTAL-1>.png (B)
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

# Output path relative to ComfyUI/output/
OUT_PREFIX_BASE = "videos/morph_test_2"

# Absolute output path (used only to write the two endpoint frames)
COMFY_OUTPUT_DIR = os.path.expanduser("~/ComfyUI/output")
ENDPOINT_DIR = os.path.join(COMFY_OUTPUT_DIR, OUT_PREFIX_BASE)

# Phase lengths
TRANS_FRAMES = 60
MID_FRAMES = 120

WAIT_FOR_EACH = True


# ---------------------------------------------------------------------
# Workflow loading / normalization
# ---------------------------------------------------------------------

ApiPrompt = Dict[str, Dict[str, Any]]


def load_as_api_prompt(path: str) -> ApiPrompt:
    """
    Load JSON and return it in ComfyUI API prompt format:
        { "<node_id>": {"class_type": "...", "inputs": {...}}, ... }

    If the file is a standard UI workflow graph ({"nodes":[...]}), this tries a
    best-effort conversion, but generally you should export/copy an API prompt.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Already in API prompt format?
    if isinstance(data, dict):
        node_like = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v and "inputs" in v
        }
        if node_like:
            return node_like

    # Best-effort conversion from a UI workflow structure
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

        # Sanity check: do we have meaningful inputs?
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

def find_node_ids(prompt: ApiPrompt) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect:
    - Latent Blend node (class_type contains both 'latent' and 'blend')
    - SaveImage node (class_type contains 'saveimage' or equals 'save image')
    """
    blend_id: Optional[str] = None
    save_id: Optional[str] = None

    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").lower()

        if save_id is None and ("saveimage" in ct or ct == "save image"):
            save_id = nid

        if blend_id is None and ("latent" in ct and "blend" in ct):
            blend_id = nid

    return blend_id, save_id


def set_blend_factor(prompt: ApiPrompt, blend_node_id: str, t: float) -> str:
    """
    Set the blend parameter on the Latent Blend node.

    Prefers common input keys first; falls back to the first numeric input found.
    Returns the key that was updated (useful for logging).
    """
    inputs = prompt[blend_node_id].setdefault("inputs", {})

    candidate_keys = [
        "blend_factor",
        "factor",
        "blend",
        "alpha",
        "t",
        "ratio",
        "mix",
        "strength",
    ]

    for key in candidate_keys:
        if key in inputs and isinstance(inputs[key], (int, float)):
            inputs[key] = float(t)
            return key

    for key, val in inputs.items():
        if isinstance(val, (int, float)):
            inputs[key] = float(t)
            return key

    raise RuntimeError("No numeric blend parameter found on the Latent Blend node.")


def set_save_prefix(prompt: ApiPrompt, save_node_id: str, prefix: str) -> None:
    """Set SaveImage filename_prefix so each frame writes to a unique file."""
    inputs = prompt[save_node_id].setdefault("inputs", {})
    inputs["filename_prefix"] = prefix


# ---------------------------------------------------------------------
# Endpoint image resolution via ComfyUI API
# ---------------------------------------------------------------------

def find_load_image_nodes(prompt: ApiPrompt) -> List[str]:
    """Return node IDs that look like LoadImage nodes."""
    ids: List[str] = []
    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").lower()
        if "loadimage" in ct or ct == "load image":
            ids.append(nid)
    return ids


def get_load_image_filename(prompt: ApiPrompt, load_node_id: str) -> str:
    """
    Extract the 'image' filename from a LoadImage node's inputs.
    This is the identifier ComfyUI uses for input images (not a filesystem path).
    """
    inputs = prompt[load_node_id].get("inputs", {})
    filename = inputs.get("image")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError(
            f"LoadImage node {load_node_id} has no valid 'image' in inputs."
        )
    return filename


def download_input_image_bytes(filename: str) -> bytes:
    """
    Fetch raw bytes of an input image from ComfyUI.
    Equivalent to downloading the exact input image as stored by ComfyUI.
    """
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
    """Poll /history/<prompt_id> until outputs exist. Returns the history entry."""
    while True:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data and data[prompt_id].get("outputs"):
            return data[prompt_id]
        time.sleep(poll_s)


def submit_prompt(job: ApiPrompt) -> str:
    """POST a prompt to ComfyUI and return its prompt_id."""
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": job}, timeout=60)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if TRANS_FRAMES < 0 or MID_FRAMES < 0:
        raise ValueError("TRANS_FRAMES and MID_FRAMES must be >= 0.")
    if MID_FRAMES < 2:
        raise ValueError("MID_FRAMES must be >= 2 to sweep blend factor 0->1.")

    total_frames = 2 * TRANS_FRAMES + MID_FRAMES

    prompt = load_as_api_prompt(WORKFLOW_PATH)

    blend_id, save_id = find_node_ids(prompt)
    if blend_id is None or save_id is None:
        print("Auto-detect failed. Available nodes (id -> class_type):")
        for nid, node in prompt.items():
            print(f"  {nid}: {node.get('class_type')}")
        raise RuntimeError(f"Auto-detect failed. blend_id={blend_id}, save_id={save_id}.")

    # Identify the two endpoint images from LoadImage nodes
    load_ids = find_load_image_nodes(prompt)
    if len(load_ids) != 2:
        print("Found LoadImage-like nodes:")
        for nid in load_ids:
            print(f"  {nid}: {prompt[nid].get('class_type')} inputs={prompt[nid].get('inputs', {})}")
        raise RuntimeError(
            f"Expected exactly 2 LoadImage nodes for endpoints, found {len(load_ids)}. "
            "If your workflow has more, hardcode which two are endpoints."
        )

    # Endpoint ordering
    load_a_id, load_b_id = load_ids[0], load_ids[1]

    file_a = get_load_image_filename(prompt, load_a_id)
    file_b = get_load_image_filename(prompt, load_b_id)

    print(f"Blend node: {blend_id}")
    print(f"Save node:  {save_id}")
    print(f"Endpoint A (LoadImage {load_a_id}): {file_a}")
    print(f"Endpoint B (LoadImage {load_b_id}): {file_b}")
    print(f"Timeline: head={TRANS_FRAMES}, mid={MID_FRAMES}, tail={TRANS_FRAMES}, total={total_frames}")

    # Write exact endpoints to output folder (byte-for-byte from ComfyUI inputs)
    os.makedirs(ENDPOINT_DIR, exist_ok=True)

    a_bytes = download_input_image_bytes(file_a)
    b_bytes = download_input_image_bytes(file_b)

    first_path = os.path.join(ENDPOINT_DIR, "frame_00000.png")
    last_path = os.path.join(ENDPOINT_DIR, f"frame_{total_frames - 1:05d}.png")

    with open(first_path, "wb") as f:
        f.write(a_bytes)
    with open(last_path, "wb") as f:
        f.write(b_bytes)

    print("Wrote endpoint frames:")
    print(f"  {first_path}")
    print(f"  {last_path}")

    # -----------------------------------------------------------------
    # Phase 1: Head (TRANS_FRAMES) at fixed t = 0.0
    # Frames: 0 .. TRANS_FRAMES-1
    # Note: frame_00000 is already written as exact bytes; we render the rest.
    # -----------------------------------------------------------------
    for frame_idx in range(1, TRANS_FRAMES):
        t = 0.0
        job: ApiPrompt = json.loads(json.dumps(prompt))
        used_key = set_blend_factor(job, blend_id, t)
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        print(f"[{frame_idx + 1:03d}/{total_frames}] head t={t:.6f} ({used_key}) prompt_id={prompt_id}")

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    # -----------------------------------------------------------------
    # Phase 2: Middle (MID_FRAMES) sweep t = 0.0 -> 1.0
    # Frames: TRANS_FRAMES .. TRANS_FRAMES + MID_FRAMES - 1
    # -----------------------------------------------------------------
    mid_start = TRANS_FRAMES
    for j in range(MID_FRAMES):
        frame_idx = mid_start + j
        t = j / (MID_FRAMES - 1)

        job: ApiPrompt = json.loads(json.dumps(prompt))
        used_key = set_blend_factor(job, blend_id, t)
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        print(f"[{frame_idx + 1:03d}/{total_frames}] mid  t={t:.6f} ({used_key}) prompt_id={prompt_id}")

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    # -----------------------------------------------------------------
    # Phase 3: Tail (TRANS_FRAMES) at fixed t = 1.0
    # Frames: TRANS_FRAMES + MID_FRAMES .. total_frames-1
    # Note: last frame is already written as exact bytes; we render up to it - 1.
    # -----------------------------------------------------------------
    tail_start = TRANS_FRAMES + MID_FRAMES
    for frame_idx in range(tail_start, total_frames - 1):
        t = 1.0
        job: ApiPrompt = json.loads(json.dumps(prompt))
        used_key = set_blend_factor(job, blend_id, t)
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{frame_idx:05d}")

        prompt_id = submit_prompt(job)
        print(f"[{frame_idx + 1:03d}/{total_frames}] tail t={t:.6f} ({used_key}) prompt_id={prompt_id}")

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    print("Done: all frames generated (endpoints are exact input bytes).")


if __name__ == "__main__":
    main()
