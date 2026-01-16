import os
import sys

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
from nuscenes.map_expansion.map_api import NuScenesMap
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- PROMPT TEMPLATE (Grounded Rewriting) ---
# We inject the {geometric_hint} to stop hallucination.
PROMPT_TEMPLATE = """
The vehicle is confirmed to be executing a **{geometric_hint}**.
Analyze the video to describe HOW this maneuver is performed visually.

**Focus strictly on:**
1. **Lane Interaction:** Does the vehicle cross a lane divider (dashed/solid)? Does it stay centered?
2. **Road Geometry:** Is the road curving, straight, or entering an intersection/junction?
3. **Dynamics:** Describe the motion relative to the road lines.

**Constraints:**
- Do NOT mention traffic lights or other cars unless they force the vehicle to stop.
- Use natural, descriptive language (e.g., "The vehicle drifts right to merge...").
- Future tense: "The ego vehicle will..."

**Output valid JSON:**
{{
  "scene_description": "[Detailed visual description of the road and maneuver]",
  "maneuver_category": "{geometric_hint}"
}}
"""

MAP_CACHE = {}

def get_nusc_map(dataroot, city):
    if city not in MAP_CACHE:
        try:
            # Assumes maps are in standard NuScenes location
            MAP_CACHE[city] = NuScenesMap(dataroot=dataroot, map_name=city)
        except:
            MAP_CACHE[city] = None
    return MAP_CACHE[city]

def get_geometric_hint(nusc_map, origin, trajectory):
    """
    Calculates the 'Physics Truth' to ground the VLM.
    Trajectory: (30, 2)
    """
    if trajectory.ndim == 3: trajectory = trajectory[0]
    
    # 1. Kinematics
    p0, p_final = trajectory[0, :2], trajectory[-1, :2]
    displacement = float(np.linalg.norm(p_final))
    y_final = float(p_final[1]) # Lateral deviation

    # Heading Change
    v_start = trajectory[5, :2] - p0
    v_end = p_final - trajectory[-6, :2]
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    diff_deg = np.degrees((angle_end - angle_start + np.pi) % (2 * np.pi) - np.pi)

    # 2. Map Context
    is_intersection = False
    if nusc_map:
        try:
            patch = (origin[0]-2, origin[1]-2, origin[0]+2, origin[1]+2)
            layers = nusc_map.get_records_in_patch(patch, ['road_segment'], mode='intersect')
            if 'road_segment' in layers:
                for t in layers['road_segment']:
                    if nusc_map.get('road_segment', t)['is_intersection']:
                        is_intersection = True; break
        except: pass
    
    context = "at an intersection" if is_intersection else "on the road"

    # 3. Categorize
    if displacement < 2.0: return "Stationary Stop"
    if abs(diff_deg) > 100: return f"U-Turn {context}"
    if diff_deg > 25: return f"Left Turn {context}"
    if diff_deg < -25: return f"Right Turn {context}"
    if y_final > 2.0: return f"Lane Change Left {context}"
    if y_final < -2.0: return f"Lane Change Right {context}"
    return f"Straight Drive {context}"

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
        """Collects CURRENT + FUTURE 6 frames."""
        tokens = [sample_token]
        cur_token = sample_token
        for _ in range(6):
            try:
                sample = self.nusc.get("sample", cur_token)
                if not sample["next"]: break # End of scene
                cur_token = sample["next"]
                tokens.append(cur_token)
            except KeyError:
                break
        return tokens

    def get_image_paths(self, sequence_tokens):
        paths = []
        for token in sequence_tokens:
            try:
                sample = self.nusc.get("sample", token)
                cam_token = sample["data"]["CAM_FRONT"]
                cam_data = self.nusc.get("sample_data", cam_token)
                full_path = os.path.join(self.nusc_dataroot, cam_data["filename"])
                paths.append(full_path)
            except: continue
        return paths

    def generate_caption(self, image_paths, geometric_hint):
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if len(valid_paths) < 2: return None

        # Format prompt with the specific hint
        formatted_prompt = PROMPT_TEMPLATE.format(geometric_hint=geometric_hint)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": valid_paths,
                    "max_pixels": 360 * 420, 
                    "fps": 2.0, 
                },
                {"type": "text", "text": formatted_prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True
                )
        except Exception as e:
            print(f"Gen Error: {e}")
            return None
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        try:
            clean_text = output_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except:
            return {"scene_description": output_text, "maneuver_category": geometric_hint}

    def process_all(self, shard_id=0, num_shards=1):
        pt_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".pt"):
                    pt_files.append(os.path.join(root, file))
        
        pt_files.sort()
        my_files = pt_files[shard_id::num_shards]
        print(f"[GPU {shard_id}] Processing {len(my_files)} files.")
        random.shuffle(my_files)

        for pt_path in tqdm(my_files, desc=f"GPU {shard_id}"):
            try:
                rel_path = os.path.relpath(pt_path, self.input_dir)
                out_path = os.path.join(self.output_dir, rel_path)
                
                if os.path.exists(out_path): continue

                # Load Data
                data = torch.load(pt_path)
                
                # --- EXTRACT GEOMETRIC HINT ---
                city = data.city if hasattr(data, 'city') else data['city']
                origin = data.origin if hasattr(data, 'origin') else data['origin']
                if hasattr(origin, 'numpy'): origin = origin.numpy()
                
                # Check for trajectory (y)
                traj = data.y if hasattr(data, 'y') else data.get('y')
                if hasattr(traj, 'cpu'): traj = traj.cpu().numpy()
                
                nusc_map = get_nusc_map(self.nusc_dataroot, city)
                
                # "The Truth"
                hint = get_geometric_hint(nusc_map, origin, traj)
                
                # --- VLM GENERATION ---
                sample_token = os.path.basename(pt_path).replace(".pt", "")
                seq_tokens = self._collect_future_sequence(sample_token)
                image_paths = self.get_image_paths(seq_tokens)
                
                caption_data = self.generate_caption(image_paths, hint)

                if caption_data:
                    # Save results
                    if isinstance(data, dict):
                        data['caption_dict'] = caption_data
                        data['scene_caption'] = caption_data.get('scene_description', '')
                    else:
                        data.caption_dict = caption_data
                        data.scene_caption = caption_data.get('scene_description', '')
                    
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    torch.save(data, out_path)

            except Exception as e:
                # print(f"Skipped {pt_path}: {e}")
                pass

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