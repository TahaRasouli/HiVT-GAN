import os
import sys
import re

# --- 1. CACHE SETUP ---
user_name = os.environ.get("USER", "rasoulta")
CACHE_DIR = f"/tmp/{user_name}/fast_cache"

try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # print(f"[SETUP] Cache forced to Local SSD: {CACHE_DIR}")
except OSError:
    CACHE_DIR = "/mount/studenten/projects/rasoulta/cache_internal" 
    # print(f"[SETUP] /tmp unreachable. Using fallback: {CACHE_DIR}")

os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["HF_HOME"] = "/mount/studenten/projects/rasoulta/cache_internal/huggingface"

import argparse
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from nuscenes.nuscenes import NuScenes
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- 2. GUIDED PROMPT (We tell it the maneuver) ---
# This forces the VLM to describe the turn, even if it looks straight initially.
PROMPT_TEMPLATE = """
You are a trajectory forecasting assistant.
The vehicle in the video is confirmed to be executing a **{geometric_hint}**.

Your task is to generate a JSON output describing this future maneuver and the road.

### GUIDELINES:
1. **Consistency:** Your description MUST match the hint: **{geometric_hint}**.
2. **Tense:** You MUST use Future Tense. Start with "The ego vehicle will..."
3. **Constraints:**
   - **NO** mention of trees, buildings, weather, or sky.
   - **NO** mention of traffic lights (use "intersection stop-line" instead).
   - **Focus** on lane lines, curbs, and dividers.

### JSON OUTPUT FORMAT:
Output a single JSON object:
{{
  "scene_description": "Natural language caption...",
  "lane_type": "Describe lane count (e.g., 2-lane one-way, 4-lane urban road, etc.)"
}}

### EXAMPLE:
**Input Hint:** Left Turn
**Output:**
{{
  "scene_description": "The ego vehicle will slow down approaching the intersection, then turn left across the perpendicular lanes to enter the target street.",
  "lane_type": "Multi-lane intersection"
}}

### YOUR TASK:
**Input Hint:** {geometric_hint}
**Output:**
"""

# --- 3. BALANCING RATES ---
# Aggressively filter Straight/Stationary to force balance
KEEP_RATES = {
    "Straight Drive": 0.15,      # Keep only 15% (Drop 85%)
    "Stationary Stop": 0.20,     # Keep only 20%
    "Left Turn": 1.0,            # Keep ALL
    "Right Turn": 1.0,           # Keep ALL
    "Lane Change Left": 1.0,     # Keep ALL
    "Lane Change Right": 1.0,    # Keep ALL
    "U-Turn": 1.0                # Keep ALL
}

def get_kinematic_category_robust(trajectory):
    """
    Robust Math Check.
    This fixes the 'Straight=U-Turn' bug by checking distance moved.
    """
    if trajectory.ndim == 3: trajectory = trajectory[0]
    if len(trajectory) < 6: return "Stationary Stop"

    p0, p_final = trajectory[0, :2], trajectory[-1, :2]
    displacement = float(np.linalg.norm(p_final))
    y_final = float(p_final[1]) 

    # 1. Stationary Check (Must move < 2m total)
    # Fixes jittery cars being called turns
    if displacement < 2.0: return "Stationary Stop"

    v_start = trajectory[5, :2] - p0
    v_end = p_final - trajectory[-6, :2]
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    diff_deg = np.degrees((angle_end - angle_start + np.pi) % (2 * np.pi) - np.pi)

    # 2. U-Turn Check (Must turn > 100 deg AND move > 4m)
    # Fixes stationary/slow cars being called U-turns
    if abs(diff_deg) > 100 and displacement > 4.0: return "U-Turn"
    
    # 3. Turn Checks
    if diff_deg > 25: return "Left Turn"
    if diff_deg < -25: return "Right Turn"
    
    # 4. Lane Change Checks (Lateral dev > 2m)
    if y_final > 2.0: return "Lane Change Left"
    if y_final < -2.0: return "Lane Change Right"
    
    return "Straight Drive"

class CaptionGenerator:
    def __init__(self, dataroot, version, input_dir, output_dir, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.nusc_dataroot = dataroot
        
        print(f"[INIT] Loading NuScenes {version}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        print(f"[INIT] Loading VLM: {model_id}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _collect_future_sequence(self, sample_token):
        tokens = [sample_token]
        cur_token = sample_token
        for _ in range(6):
            try:
                sample = self.nusc.get("sample", cur_token)
                if not sample["next"]: break
                cur_token = sample["next"]
                tokens.append(cur_token)
            except KeyError: break
        return tokens

    def get_image_paths(self, sequence_tokens):
        paths = []
        for token in sequence_tokens:
            try:
                sample = self.nusc.get("sample", token)
                cam_token = sample["data"]["CAM_FRONT"]
                cam_data = self.nusc.get("sample_data", cam_token)
                paths.append(os.path.join(self.nusc_dataroot, cam_data["filename"]))
            except: continue
        return paths

    def generate_caption(self, image_paths, geometric_hint):
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if len(valid_paths) < 2: return None

        # Pass the robust math hint to the prompt
        formatted_prompt = PROMPT_TEMPLATE.format(geometric_hint=geometric_hint)

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": valid_paths, "max_pixels": 360 * 420, "fps": 2.0},
                {"type": "text", "text": formatted_prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=200, do_sample=False, num_beams=1
                )
        except Exception:
            return None
        
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        try:
            # Robust Cleaning
            cleaned = re.sub(r'```json\s*', '', output_text, flags=re.IGNORECASE)
            cleaned = cleaned.replace('```', '').strip()
            cleaned = cleaned.replace('{{', '{').replace('}}', '}')
            
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1:
                return json.loads(cleaned[start : end+1])
            else:
                return None
        except:
            return None

    def process_all(self, shard_id=0, num_shards=1):
        print(f"[BALANCING] Rates: {KEEP_RATES}")
        
        pt_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".pt"):
                    pt_files.append(os.path.join(root, file))
        
        pt_files.sort()
        random.Random(42).shuffle(pt_files) 
        my_files = pt_files[shard_id::num_shards]
        print(f"[GPU {shard_id}] Scanning {len(my_files)} files...")

        # Count using the MATH Ground Truth (which we force the VLM to use)
        final_stats = {k: 0 for k in KEEP_RATES.keys()}
        
        for pt_path in tqdm(my_files, desc=f"GPU {shard_id}"):
            try:
                rel_path = os.path.relpath(pt_path, self.input_dir)
                out_path = os.path.join(self.output_dir, rel_path)
                if os.path.exists(out_path): continue

                data = torch.load(pt_path, weights_only=False)
                traj = data.y if hasattr(data, 'y') else data.get('y')
                if hasattr(traj, 'cpu'): traj = traj.cpu().numpy()

                # 1. CALCULATE GROUND TRUTH (MATH)
                math_category = get_kinematic_category_robust(traj)
                
                # 2. FILTER
                keep_rate = KEEP_RATES.get(math_category, 1.0)
                if random.random() > keep_rate:
                    continue 

                # 3. GENERATE (Using Hint)
                sample_token = os.path.basename(pt_path).replace(".pt", "")
                seq_tokens = self._collect_future_sequence(sample_token)
                image_paths = self.get_image_paths(seq_tokens)
                
                if not image_paths: continue

                # We pass 'math_category' to the VLM
                caption_data = self.generate_caption(image_paths, math_category)

                if caption_data:
                    # 4. SAVE (We force the Math Label into the data)
                    # This ensures the label 'maneuver_category' is 100% correct geometrically
                    # while the text description tries its best to describe it.
                    caption_data['maneuver_category'] = math_category
                    
                    if isinstance(data, dict):
                        data['caption_dict'] = caption_data
                        data['scene_caption'] = caption_data.get('scene_description', '')
                        data['maneuver_category'] = math_category
                    else:
                        data.caption_dict = caption_data
                        data.scene_caption = caption_data.get('scene_description', '')
                        data.maneuver_category = math_category
                    
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    torch.save(data, out_path)
                    
                    # Log
                    if math_category in final_stats: 
                        final_stats[math_category] += 1
                    else:
                        final_stats[math_category] = final_stats.get(math_category, 0) + 1

            except Exception:
                pass
        
        print(f"[GPU {shard_id}] Final Distribution: {final_stats}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    gen = CaptionGenerator(
        dataroot=args.dataroot,
        version=args.version,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    gen.process_all(shard_id=args.shard_id, num_shards=args.num_shards)

if __name__ == "__main__":
    main()