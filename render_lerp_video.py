import json
import time
import os
import requests
from typing import Any, Dict, Tuple, Optional, List

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "cursed1.json"

# Where ComfyUI will save generated frames (relative to ComfyUI output/)
OUT_PREFIX_BASE = "videos/morph_test_1"

# Where we will ALSO write exact endpoints on disk (absolute path under ComfyUI output/)
COMFY_OUTPUT_DIR = os.path.expanduser("~/ComfyUI/output")
ENDPOINT_DIR = os.path.join(COMFY_OUTPUT_DIR, OUT_PREFIX_BASE)

FRAMES = 120
WAIT_FOR_EACH = True

# -----------------------------------------------------------------------------
# Loading / format normalization
# -----------------------------------------------------------------------------

def load_as_api_prompt(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # API prompt format: dict of node_id -> {"class_type":..., "inputs":...}
    if isinstance(data, dict):
        node_like = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and "class_type" in v and "inputs" in v
        }
        if node_like:
            return node_like

    # Standard workflow format: {"nodes":[...], ...} (best-effort only)
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        prompt: Dict[str, Dict[str, Any]] = {}
        for n in data["nodes"]:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            class_type = n.get("type") or n.get("class_type")
            inputs = n.get("inputs", {})
            if nid is None or not class_type:
                continue
            prompt[str(nid)] = {"class_type": class_type, "inputs": inputs}

        if prompt and any(isinstance(v.get("inputs"), dict) and len(v["inputs"]) > 0 for v in prompt.values()):
            return prompt

        raise RuntimeError(
            "Workflow JSON does not contain sufficient API 'inputs'. "
            "You must export/copy an API prompt JSON (node_id -> {class_type, inputs})."
        )

    raise RuntimeError("Unrecognized workflow file format.")

# -----------------------------------------------------------------------------
# Node detection / mutation
# -----------------------------------------------------------------------------

def find_node_ids(prompt: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    blend_id = None
    save_id = None

    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").lower()

        if save_id is None and ("saveimage" in ct or ct == "save image"):
            save_id = nid

        # Match: "Latent Blend" (your node)
        if blend_id is None and ("latent" in ct and "blend" in ct):
            blend_id = nid

    return blend_id, save_id

def set_blend_factor(prompt: Dict[str, Dict[str, Any]], blend_node_id: str, t: float) -> str:
    """
    Set the blend parameter on the Latent Blend node.
    Prefers the common ComfyUI input name: 'blend_factor'
    Falls back to other common names.
    """
    inputs = prompt[blend_node_id].setdefault("inputs", {})

    preferred_keys = [
        "blend_factor",   # shown in your screenshot
        "factor",
        "blend",
        "alpha",
        "t",
        "ratio",
        "mix",
        "strength",
    ]

    for key in preferred_keys:
        if key in inputs and isinstance(inputs[key], (int, float)):
            inputs[key] = float(t)
            return key

    # Last resort: first numeric input
    for key, val in inputs.items():
        if isinstance(val, (int, float)):
            inputs[key] = float(t)
            return key

    raise RuntimeError(
        "Could not find a numeric blend parameter to set on the Latent Blend node."
    )

def set_save_prefix(prompt: Dict[str, Dict[str, Any]], save_node_id: str, prefix: str) -> None:
    inputs = prompt[save_node_id].setdefault("inputs", {})
    inputs["filename_prefix"] = prefix

# -----------------------------------------------------------------------------
# Fetch exact input images via API (no explicit filesystem paths)
# -----------------------------------------------------------------------------

def find_load_image_nodes(prompt: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return node IDs that look like LoadImage nodes."""
    load_ids = []
    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").lower()
        if "loadimage" in ct or ct == "load image":
            load_ids.append(nid)
    return load_ids

def get_load_image_filename(prompt: Dict[str, Dict[str, Any]], load_node_id: str) -> str:
    inputs = prompt[load_node_id].get("inputs", {})
    filename = inputs.get("image")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError(f"LoadImage node {load_node_id} does not have an 'image' filename in inputs.")
    return filename

def download_input_image_bytes(filename: str) -> bytes:
    r = requests.get(f"{COMFY_URL}/view", params={"filename": filename, "type": "input"}, timeout=60)
    r.raise_for_status()
    return r.content

# -----------------------------------------------------------------------------
# ComfyUI API helpers
# -----------------------------------------------------------------------------

def wait_for_prompt_done(prompt_id: str, poll_s: float = 0.5) -> Dict[str, Any]:
    while True:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data and data[prompt_id].get("outputs"):
            return data[prompt_id]
        time.sleep(poll_s)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    prompt = load_as_api_prompt(WORKFLOW_PATH)

    blend_id, save_id = find_node_ids(prompt)
    if blend_id is None or save_id is None:
        print("Auto-detect failed. Available nodes (id -> class_type):")
        for nid, node in prompt.items():
            print(f"  {nid}: {node.get('class_type')}")
        raise RuntimeError(f"Auto-detect failed. blend_id={blend_id}, save_id={save_id}.")

    # --- Resolve the two endpoint images from LoadImage nodes ---
    load_ids = find_load_image_nodes(prompt)
    if len(load_ids) != 2:
        print("Found LoadImage-like nodes:")
        for nid in load_ids:
            print(f"  {nid}: {prompt[nid].get('class_type')} inputs={prompt[nid].get('inputs', {})}")
        raise RuntimeError(
            f"Expected exactly 2 LoadImage nodes for endpoints, found {len(load_ids)}. "
            "If your workflow has more, hardcode which two are endpoints by selecting their node IDs."
        )

    # Keep your previous swap (if you need it). If not, set load_a_id=load_ids[0], load_b_id=load_ids[1].
    load_a_id, load_b_id = load_ids[0], load_ids[1]

    file_a = get_load_image_filename(prompt, load_a_id)
    file_b = get_load_image_filename(prompt, load_b_id)

    print(f"Using blend node id: {blend_id}")
    print(f"Using save node id: {save_id}")
    print(f"Endpoint A from LoadImage node {load_a_id}: {file_a}")
    print(f"Endpoint B from LoadImage node {load_b_id}: {file_b}")

    # --- Write exact endpoints (byte-for-byte) into output folder ---
    os.makedirs(ENDPOINT_DIR, exist_ok=True)

    a_bytes = download_input_image_bytes(file_a)
    b_bytes = download_input_image_bytes(file_b)

    first_path = os.path.join(ENDPOINT_DIR, "frame_00000.png")
    last_path  = os.path.join(ENDPOINT_DIR, f"frame_{FRAMES-1:05d}.png")

    with open(first_path, "wb") as f:
        f.write(a_bytes)
    with open(last_path, "wb") as f:
        f.write(b_bytes)

    print(f"Wrote exact endpoint frames:\n  {first_path}\n  {last_path}")

    # --- Render only middle frames with ComfyUI ---
    for i in range(1, FRAMES - 1):
        t = i / (FRAMES - 1)

        job = json.loads(json.dumps(prompt))

        used_key = set_blend_factor(job, blend_id, t)

        # Save middle frames into the same folder prefix.
        set_save_prefix(job, save_id, f"{OUT_PREFIX_BASE}/frame_{i:05d}")

        resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": job}, timeout=60)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]

        print(f"[{i+1:03d}/{FRAMES}] blend={t:.6f} ({used_key}) prompt_id={prompt_id}")

        if WAIT_FOR_EACH:
            wait_for_prompt_done(prompt_id)

    print("All frames generated (endpoints are exact input bytes).")

if __name__ == "__main__":
    main()
