import os
import sys
import re

# --- 1. CACHE SETUP ---
user_name = os.environ.get("USER", "rasoulta")
CACHE_DIR = f"/tmp/{user_name}/fast_cache"

try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[SETUP] Cache forced to Local SSD: {CACHE_DIR}")
except OSError:
    CACHE_DIR = "/mount/studenten/projects/rasoulta/cache_internal" 
    print(f"[SETUP] /tmp unreachable. Using fallback: {CACHE_DIR}")

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

# --- 2. FIXED PROMPT (Single Braces) ---
PROMPT_TEMPLATE = """
You are a trajectory forecasting assistant. Analyze the video to describe the future motion of the ego vehicle and the road configuration.

### GUIDELINES:
1. **Action:** Describe the maneuver (Straight, Turn, Lane Change, Stop) based STRICTLY on the video.
2. **Tense:** Use Future Tense. Start with "The ego vehicle will..."
3. **Constraints:**
   - **NO** mention of trees, buildings, weather, or sky.
   - **NO** mention of traffic lights (use "intersection stop-line" instead).
   - **Focus** on lane lines, curbs, and dividers.

### JSON OUTPUT FORMAT:
You must output a single JSON object with these 3 keys:
- "scene_description": The natural language caption.
- "maneuver_category": One of [Straight Drive, Turn Left, Turn Right, U-Turn, Lane Change Left, Lane Change Right, Stationary].
- "lane_type": Describe the lane count/type (e.g., "Single-lane road", "2-lane one-way", "3-lane highway", "Multi-lane intersection").

### EXAMPLE:
**Output:**
{
  "scene_description": "The ego vehicle will maintain a steady pace, staying centered in the rightmost lane while passing an intersection.",
  "maneuver_category": "Straight Drive",
  "lane_type": "4-lane road"
}

### YOUR TASK:
**Output:**
"""

# --- 3. BALANCING ---
KEEP_RATES = {
    "Straight Drive": 0.2,       
    "Stationary Stop": 0.25,     
    "Left Turn": 1.0,            
    "Right Turn": 1.0,           
    "Lane Change Left": 1.0,     
    "Lane Change Right": 1.0,    
    "U-Turn": 1.0                
}

def get_kinematic_category_robust(trajectory):
    if trajectory.ndim == 3: trajectory = trajectory[0]
    if len(trajectory) < 6: return "Stationary Stop"

    p0, p_final = trajectory[0, :2], trajectory[-1, :2]
    displacement = float(np.linalg.norm(p_final))
    y_final = float(p_final[1]) 

    if displacement < 2.0: return "Stationary Stop"

    v_start = trajectory[5, :2] - p0
    v_end = p_final - trajectory[-6, :2]
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    diff_deg = np.degrees((angle_end - angle_start + np.pi) % (2 * np.pi) - np.pi)

    if abs(diff_deg) > 100 and displacement > 4.0: return "U-Turn"
    if diff_deg > 25: return "Left Turn"
    if diff_deg < -25: return "Right Turn"
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

    def generate_caption(self, image_paths):
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if len(valid_paths) < 2: return None

        formatted_prompt = PROMPT_TEMPLATE 

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
                    **inputs, 
                    max_new_tokens=200, 
                    do_sample=False, 
                    num_beams=1
                )
        except Exception as e:
            print(f"VLM Internal Error: {e}")
            return None
        
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # --- IMPROVED PARSER ---
        try:
            # 1. Clean Markdown and Double Braces
            cleaned = re.sub(r'```json\s*', '', output_text, flags=re.IGNORECASE)
            cleaned = cleaned.replace('```', '').strip()
            # FIX: Handle double braces if they still appear
            cleaned = cleaned.replace('{{', '{').replace('}}', '}')
            
            # 2. Extract JSON
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            
            if start != -1 and end != -1:
                json_str = cleaned[start : end+1]
                return json.loads(json_str)
            else:
                return None
        except Exception:
            # Fallback: Capture raw text so we don't lose the sample
            # (We can clean up VLM_RAW_OUTPUT later)
            return {
                "scene_description": output_text.replace('\n', ' ').strip(),
                "maneuver_category": "VLM_RAW_OUTPUT", 
                "lane_type": "Unknown"
            }

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

        final_stats = {k: 0 for k in KEEP_RATES.keys()}
        final_stats["VLM_RAW_OUTPUT"] = 0
        
        for pt_path in tqdm(my_files, desc=f"GPU {shard_id}"):
            try:
                rel_path = os.path.relpath(pt_path, self.input_dir)
                out_path = os.path.join(self.output_dir, rel_path)
                if os.path.exists(out_path): continue

                data = torch.load(pt_path, weights_only=False)
                
                traj = data.y if hasattr(data, 'y') else data.get('y')
                if hasattr(traj, 'cpu'): traj = traj.cpu().numpy()

                filter_category = get_kinematic_category_robust(traj)
                
                keep_rate = KEEP_RATES.get(filter_category, 1.0)
                if random.random() > keep_rate:
                    continue 

                sample_token = os.path.basename(pt_path).replace(".pt", "")
                seq_tokens = self._collect_future_sequence(sample_token)
                image_paths = self.get_image_paths(seq_tokens)
                
                if not image_paths: continue

                caption_data = self.generate_caption(image_paths)

                if caption_data:
                    # Save structure
                    if isinstance(data, dict):
                        data['caption_dict'] = caption_data
                        data['scene_caption'] = caption_data.get('scene_description', '')
                        data['maneuver_category'] = caption_data.get('maneuver_category', 'Unknown')
                        data['lane_type'] = caption_data.get('lane_type', 'Unknown')
                    else:
                        data.caption_dict = caption_data
                        data.scene_caption = caption_data.get('scene_description', '')
                        data.maneuver_category = caption_data.get('maneuver_category', 'Unknown')
                        data.lane_type = caption_data.get('lane_type', 'Unknown')
                    
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    torch.save(data, out_path)
                    
                    # Stats
                    vlm_cat = caption_data.get('maneuver_category', 'Unknown')
                    stats_key = "Unknown"
                    
                    if vlm_cat == "VLM_RAW_OUTPUT":
                        stats_key = "VLM_RAW_OUTPUT"
                    else:
                        for k in final_stats.keys():
                            if k.split()[0] in vlm_cat: 
                                stats_key = k; break
                    
                    if stats_key in final_stats: final_stats[stats_key] += 1
                    else: final_stats[stats_key] = final_stats.get(stats_key, 0) + 1

            except Exception:
                pass
        
        print(f"[GPU {shard_id}] Final Counts: {final_stats}")

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