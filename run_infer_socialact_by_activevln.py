#!/usr/bin/env python3
"""
SocialACT Dataset Offline Evaluation using ActiveVLN

Connects to a vllm OpenAI-compatible endpoint (the original ActiveVLN design).
Prefix caching on the vllm side handles KV cache reuse across turns — no
linear slowdown with growing conversation history.

For each sequence (Verbal_###, Nonverbal_###):
  1. Load instruction from per-sequence JSON file
  2. Iterate all frames (front image, 2 Hz) sequentially
  3. Run ActiveVLN multi-turn inference via API
  4. Save per-sequence CSV (predictions) + MP4 (visualization)

Usage:
    # 1. Start vllm server (separate terminal):
    #   vllm serve /path/to/model --task generate --trust-remote-code \
    #       --limit-mm-per-prompt image=200,video=0 \
    #       --mm_processor_kwargs '{"max_pixels": 80000}' \
    #       --max-model-len 32768 --enable-prefix-caching \
    #       --disable-log-requests --port 8003
    #
    # 2. Run evaluation:
    #   export OPENAI_API_KEY="EMPTY"
    #   export OPENAI_API_BASE="http://127.0.0.1:8003/v1"
    #   python run_infer_socialact_by_activevln.py
"""

import os
import sys
import json
import re
import csv
import time
import base64
import glob
import argparse
import numpy as np
import cv2
import matplotlib.cm as mpl_cm
from io import BytesIO
from PIL import Image
from openai import OpenAI

try:
    from qwen_vl_utils.vision_process import smart_resize
except ImportError:
    def smart_resize(height, width, max_pixels=76800, factor=28):
        scale = (max_pixels / (height * width)) ** 0.5
        if scale < 1.0:
            new_h = max(factor, round(int(height * scale) / factor) * factor)
            new_w = max(factor, round(int(width  * scale) / factor) * factor)
            return new_h, new_w
        return height, width


# ---------------------------------------------------------------------------
#  Prompts — identical to eval_vlnce.py (R2R action space)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), "
    "or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, "
    "move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, "
    "turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. "
    "You can take multiple actions at each turn. "
)

FIRST_TURN_PROMPT = (
    "Instruction: {}"
    "Decide your next action. "
    "You can take up to 3 actions at a time, separated by ','. "
)

NORMAL_PROMPT = (
    "Instruction: {}"
    "Decide your next action. "
    "You can take up to 3 actions at a time, separated by ','. "
)


# ---------------------------------------------------------------------------
#  Dataset helpers
# ---------------------------------------------------------------------------

def encode_pil_to_base64(pil_image: Image.Image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_instruction(json_path: str):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("Instruction", None)
    except Exception as e:
        print(f"[WARN] Cannot load {json_path}: {e}")
        return None


def get_frame_indices(seq_dir: str) -> list:
    dirs = sorted(glob.glob(os.path.join(seq_dir, "frame_*")))
    indices = []
    for d in dirs:
        try:
            indices.append(int(os.path.basename(d).split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(indices)


def load_frame_images(seq_dir: str, frame_idx: int):
    """Load front/left/right as RGB numpy arrays, or (None,None,None) on failure."""
    name = f"frame_{frame_idx:04d}"
    frame_dir = os.path.join(seq_dir, name)
    result = {}
    for view in ("front", "left", "right"):
        path = os.path.join(frame_dir, f"{name}_{view}.jpg")
        if not os.path.exists(path):
            return None, None, None
        bgr = cv2.imread(path)
        if bgr is None:
            return None, None, None
        result[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return result["front"], result["left"], result["right"]


# ---------------------------------------------------------------------------
#  Visualisation helpers
# ---------------------------------------------------------------------------

def draw_action_on_front(
    img: np.ndarray,
    parsed_actions: list,
    arrow_thickness: int = 2,
    tip_length: float = 0.40,
    stop_radius: int = 10,
    arrow_scale: float = 0.14,
    vis_scale: float = 120.0,
    slot_gap: int = 6,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    base_x, base_y = w // 2, int(h * 0.92)

    try:
        cmap = mpl_cm.get_cmap("turbo")
    except Exception:
        cmap = mpl_cm.get_cmap("viridis")

    direction_map = {
        "forward":    ( 0.0,  1.0),
        "turn_left":  (-0.7,  0.7),
        "turn_right": ( 0.7,  0.7),
    }
    slot_h = max(1, int(vis_scale * arrow_scale) + slot_gap)
    n = len(parsed_actions)

    for i, action in enumerate(parsed_actions):
        atype = action.get("type", "")
        t = (i + 0.5) / n if n > 0 else 0.5
        rgba = cmap(t)[:3]
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

        sx, sy = base_x, base_y - i * slot_h
        start = (np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1))

        if atype == "stop":
            cv2.circle(out, start, stop_radius, (0, 0, 220), -1)
            cv2.circle(out, start, stop_radius, (255, 255, 255), 1)
            cv2.putText(out, "STOP", (start[0] - 18, start[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            continue

        dx_dir, dy_dir = direction_map.get(atype, (0.0, 1.0))
        ex = int(sx + dx_dir * arrow_scale * vis_scale)
        ey = int(sy - dy_dir * arrow_scale * vis_scale)
        end = (np.clip(ex, 0, w - 1), np.clip(ey, 0, h - 1))

        if np.linalg.norm(np.array(end) - np.array(start)) > 2:
            cv2.arrowedLine(out, start, end, color, arrow_thickness,
                            tipLength=tip_length, line_type=cv2.LINE_AA)
    return out


def add_instruction_bar(img_rgb: np.ndarray, display_text: str,
                         bar_height: int = 80) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    canvas = np.full((h + bar_height, w, 3), 255, dtype=np.uint8)
    canvas[:h] = img_rgb

    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (_, line_h), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
    margin_x, margin_y = 10, 10
    max_line_w = w - 2 * margin_x

    words, lines, line = (display_text or "").split(), [], ""
    for word in words:
        test = (line + " " + word) if line else word
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_line_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = h + margin_y + line_h
    for i, ln in enumerate(lines[:4]):
        dy = y + i * (line_h + 4)
        if dy > h + bar_height - 5:
            break
        cv2.putText(canvas, ln, (margin_x, dy),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
#  ActiveVLN Agent — vllm OpenAI-compatible endpoint (eval_vlnce.py style)
# ---------------------------------------------------------------------------

class ActiveVLNAgent:

    def __init__(self, client: OpenAI, model_id: str,
                 max_pixels: int = 76800, max_turns: int = 200):
        self.client     = client
        self.model_id   = model_id
        self.max_pixels = max_pixels
        self.max_turns  = max_turns
        self.sampling_params = {"temperature": 0.2, "max_tokens": 512, "top_p": 0.8}
        self.conversations: list = []
        self.count_turn: int     = 0
        self._init_conversation()

    def _init_conversation(self):
        self.conversations = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }]
        self.count_turn = 0

    def reset(self):
        self._init_conversation()

    def _preprocess(self, rgb_np: np.ndarray) -> str:
        """Resize image and return base64 JPEG string (matches eval_vlnce.py)."""
        pil = Image.fromarray(rgb_np.astype("uint8")).convert("RGB")
        new_h, new_w = smart_resize(pil.height, pil.width,
                                    max_pixels=self.max_pixels, factor=28)
        pil = pil.resize((new_w, new_h))
        return encode_pil_to_base64(pil)

    def infer(self, rgb_np: np.ndarray, instruction: str) -> str:
        """One inference step — mirrors eval_vlnce.py ActiveVlnAgent.act()."""
        if self.count_turn >= self.max_turns:
            return "stop"

        b64 = self._preprocess(rgb_np)
        is_first   = (len(self.conversations) == 1)
        obs_prefix = "[Initial Observation]:" if is_first else "After that, the observation is:"
        prompt     = (FIRST_TURN_PROMPT if is_first else NORMAL_PROMPT).format(instruction)

        content = [
            {"type": "text",      "text": obs_prefix},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text",      "text": prompt},
        ]
        self.conversations.append({"role": "user", "content": content})

        try:
            resp = self.client.chat.completions.create(
                messages=self.conversations,
                model=self.model_id,
                max_completion_tokens=self.sampling_params["max_tokens"],
                temperature=self.sampling_params["temperature"],
                top_p=self.sampling_params["top_p"],
            )
            output = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            output = "stop"

        self.conversations.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output}],
        })
        self.count_turn += 1
        return output

    @staticmethod
    def parse_actions(output: str) -> list:
        results = []
        for part in output.split(","):
            part = part.strip().lower()
            if "stop" in part:
                results.append({"type": "stop"})
            elif "forward" in part:
                m = re.search(r"(\d+)", part)
                results.append({"type": "forward",    "value_cm":  int(m.group(1)) if m else 25})
            elif "left" in part:
                m = re.search(r"(\d+)", part)
                results.append({"type": "turn_left",  "value_deg": int(m.group(1)) if m else 15})
            elif "right" in part:
                m = re.search(r"(\d+)", part)
                results.append({"type": "turn_right", "value_deg": int(m.group(1)) if m else 15})
        return results


# ---------------------------------------------------------------------------
#  Evaluator
# ---------------------------------------------------------------------------

class ActiveVLNSocialACTEvaluator:

    def __init__(self, api_key: str, base_url: str,
                 max_pixels: int = 76800, max_turns: int = 200):
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            model_id = client.models.list().data[0].id
        except Exception as e:
            print(f"\n[ERROR] Cannot connect to vllm endpoint: {base_url}")
            print(f"        Start the server first:")
            print(f"          vllm serve $MODEL_PATH --task generate --trust-remote-code \\")
            print(f"              --limit-mm-per-prompt image=200,video=0 \\")
            print(f"              --mm_processor_kwargs '{{\"max_pixels\": 80000}}' \\")
            print(f"              --max-model-len 32768 --enable-prefix-caching \\")
            print(f"              --disable-log-requests --port 8003")
            print(f"        Detail: {e}\n")
            sys.exit(1)
        print(f"[ActiveVLN] Model: {model_id}  |  Endpoint: {base_url}")
        self.agent = ActiveVLNAgent(client, model_id,
                                    max_pixels=max_pixels, max_turns=max_turns)

    # ------------------------------------------------------------------

    def evaluate_sequence(self, seq_name: str, seq_dir: str,
                           instruction: str, output_dir: str):
        frame_indices = get_frame_indices(seq_dir)
        if not frame_indices:
            print(f"[WARN] No frames in {seq_dir}, skipping.")
            return

        print(f"\n{'*' * 60}")
        print(f"  SEQ  : {seq_name}  ({len(frame_indices)} frames)")
        print(f"  Instr: {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        print(f"{'*' * 60}")

        self.agent.reset()

        csv_records   = []
        vis_frames    = []
        vis_labels    = []
        total_time    = 0.0
        display_label = f"Instruction: {instruction}"

        for frame_idx in frame_indices:
            front, left, right = load_frame_images(seq_dir, frame_idx)
            if front is None:
                print(f"  [WARN] Missing images for frame_{frame_idx:04d}, skipping")
                continue

            t0 = time.time()
            raw_output = self.agent.infer(front, instruction)
            elapsed = time.time() - t0
            total_time += elapsed

            parsed = ActiveVLNAgent.parse_actions(raw_output)
            action_types_str = "|".join(a["type"] for a in parsed) if parsed else "none"

            print(f"  [frame_{frame_idx:04d}]  {elapsed:.3f}s  → {raw_output}")

            csv_records.append({
                "frame_idx":    frame_idx,
                "raw_action":   raw_output,
                "num_actions":  len(parsed),
                "action_types": action_types_str,
                "infer_time_s": round(elapsed, 4),
            })

            front_vis = draw_action_on_front(front, parsed)
            vis_frames.append((left, front_vis, right))
            vis_labels.append(display_label)

        os.makedirs(output_dir, exist_ok=True)
        self._save_csv(csv_records, output_dir, seq_name)
        self._save_video(vis_frames, vis_labels, output_dir, seq_name)

        n = len(csv_records)
        if n > 0:
            print(f"  [{seq_name}] {n} frames | avg {total_time/n:.3f}s/frame")

    # ------------------------------------------------------------------

    @staticmethod
    def _save_csv(records: list, output_dir: str, seq_name: str):
        path = os.path.join(output_dir, f"{seq_name}.csv")
        fields = ["frame_idx", "raw_action", "num_actions", "action_types", "infer_time_s"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f"  CSV  → {path}  ({len(records)} rows)")

    @staticmethod
    def _save_video(frames: list, instructions: list, output_dir: str, seq_name: str):
        """Save MP4: [LEFT | FRONT(arrows) | RIGHT] + instruction bar, 2 fps."""
        if not frames:
            return
        path = os.path.join(output_dir, f"{seq_name}.mp4")

        _, front0, _ = frames[0]
        target_h, target_w = front0.shape[:2]
        bar_h = 80
        out_w = target_w * 3
        out_h = target_h + bar_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 2.0, (out_w, out_h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), 2.0, (out_w, out_h))
        if not writer.isOpened():
            print(f"  [ERROR] Cannot create video: {path}")
            return

        def _resize(img):
            if img.shape[:2] != (target_h, target_w):
                return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return img

        for i, (lf, ff, rf) in enumerate(frames):
            combined  = np.concatenate([_resize(lf), _resize(ff), _resize(rf)], axis=1)
            label     = instructions[i] if i < len(instructions) else ""
            frame_bar = add_instruction_bar(combined, label, bar_h)
            writer.write(cv2.cvtColor(frame_bar, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"  MP4  → {path}  ({len(frames)} frames, {out_w}x{out_h}, 2 fps)")

    # ------------------------------------------------------------------

    def evaluate_all(self, dataset_dir: str, instruction_dir: str, output_dir: str):
        seqs = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
            and (d.startswith("Verbal_") or d.startswith("Nonverbal_"))
        ])
        print(f"\n[ActiveVLN] Found {len(seqs)} sequences in {dataset_dir}")
        print(f"[ActiveVLN] Instructions : {instruction_dir}")
        print(f"[ActiveVLN] Output       : {os.path.abspath(output_dir)}\n")

        if not seqs:
            print("[ERROR] No Verbal_*/Nonverbal_* folders found. Exiting.")
            return

        for i, seq_name in enumerate(seqs):
            csv_out = os.path.join(output_dir, f"{seq_name}.csv")
            mp4_out = os.path.join(output_dir, f"{seq_name}.mp4")
            if os.path.exists(csv_out) and os.path.exists(mp4_out):
                print(f"[SKIP] {seq_name}: already completed")
                continue

            json_path = os.path.join(instruction_dir, f"{seq_name}.json")
            if not os.path.exists(json_path):
                print(f"[WARN] No JSON for {seq_name}, skipping")
                continue

            instruction = load_instruction(json_path)
            if not instruction:
                print(f"[WARN] Empty instruction for {seq_name}, skipping")
                continue

            print(f"\n{'#' * 60}")
            print(f"# [{i + 1}/{len(seqs)}] {seq_name}")
            print(f"{'#' * 60}")

            self.evaluate_sequence(
                seq_name,
                os.path.join(dataset_dir, seq_name),
                instruction,
                output_dir,
            )

        print(f"\n[ActiveVLN] All sequences evaluated.")
        print(f"[ActiveVLN] Results → {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline SocialACT evaluation using ActiveVLN via vllm endpoint")
    parser.add_argument("--dataset-dir",
        default=os.environ.get("DATASET_DIR",
            "/home/aprl/Desktop/when2reason/OmniNav_setup/SocialACT_revision/"
            "07_New_cube_and_panorama_images_2hz/Labeled"))
    parser.add_argument("--instruction-dir",
        default=os.environ.get("INSTR_DIR",
            "/home/aprl/Desktop/when2reason/OmniNav_setup/SocialACT_revision/"
            "05_Refactored_data/High-level_Instructions/Labeled_v1"))
    parser.add_argument("--output-dir",
        default=os.environ.get("OUTPUT_DIR",
            "experiment_results/ActiveVLN/SocialACT_High_Labeled_v1"))
    parser.add_argument("--mission", default=None,
        help="Run a single mission only (e.g. Nonverbal_001). Omit to run all.")
    parser.add_argument("--max-pixels", type=int, default=76800)
    parser.add_argument("--max-turns",  type=int, default=200)
    args = parser.parse_args()

    api_key  = os.environ.get("OPENAI_API_KEY", "EMPTY")
    base_url = os.environ.get("OPENAI_API_BASE")
    if not base_url:
        print("[ERROR] OPENAI_API_BASE is not set.")
        print("        export OPENAI_API_BASE=http://127.0.0.1:8003/v1")
        sys.exit(1)

    for path, label in [
        (args.dataset_dir,     "dataset-dir"),
        (args.instruction_dir, "instruction-dir"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)

    evaluator = ActiveVLNSocialACTEvaluator(
        api_key=api_key,
        base_url=base_url,
        max_pixels=args.max_pixels,
        max_turns=args.max_turns,
    )

    print("\n" + "=" * 80)
    print("  SOCIALACT OFFLINE EVALUATION — ActiveVLN")
    print("=" * 80)

    if args.mission:
        seq_dir   = os.path.join(args.dataset_dir, args.mission)
        json_path = os.path.join(args.instruction_dir, f"{args.mission}.json")
        if not os.path.isdir(seq_dir):
            print(f"[ERROR] Mission directory not found: {seq_dir}")
            sys.exit(1)
        if not os.path.exists(json_path):
            print(f"[ERROR] Instruction JSON not found: {json_path}")
            sys.exit(1)
        instruction = load_instruction(json_path)
        if not instruction:
            print(f"[ERROR] Empty instruction in {json_path}")
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)
        evaluator.evaluate_sequence(args.mission, seq_dir, instruction, args.output_dir)
    else:
        evaluator.evaluate_all(args.dataset_dir, args.instruction_dir, args.output_dir)

    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETE")
    print(f"  Results → {os.path.abspath(args.output_dir)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
