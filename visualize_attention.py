import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
import sys
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# --- CONFIGURATION ---
CKPT_PATH = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-GAN/lightning_logs/version_54/checkpoints/epoch=29-step=8040.ckpt"
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
NUSCENES_MAP_ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta"

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
except ImportError:
    print("Error: nuscenes-devkit not installed.")
    sys.exit(1)

MAP_CACHE = {}

def get_map_features(city, origin, theta, radius=60):
    if city not in MAP_CACHE:
        try:
            MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
        except Exception as e:
            return {}, {}

    nusc_map = MAP_CACHE[city]
    x, y = origin[0], origin[1]
    patch_box = (x - radius, y - radius, x + radius, y + radius)
    
    # FETCH VISUAL LAYERS (Dividers = Paint Lines)
    layers = ['lane_divider', 'road_divider', 'ped_crossing']
    records = nusc_map.get_records_in_patch(patch_box, layers, mode='intersect')
    
    features = {'dividers': [], 'crossings': []}
    
    # 1. Process Dividers (White Lines)
    for layer in ['lane_divider', 'road_divider']:
        for token in records.get(layer, []):
            try:
                line = nusc_map.get(layer, token)
                nodes = [nusc_map.get('node', t) for t in line['line_token']]
                points = np.array([[n['x'], n['y']] for n in nodes])
                features['dividers'].append(transform_to_local(points, x, y, theta))
            except: continue

    # 2. Process Crossings (Context)
    for token in records.get('ped_crossing', []):
        try:
            poly = nusc_map.get('ped_crossing', token)
            nodes = [nusc_map.get('node', t) for t in poly['exterior_node_tokens']]
            points = np.array([[n['x'], n['y']] for n in nodes])
            features['crossings'].append(transform_to_local(points, x, y, theta))
        except: continue
        
    return features

def transform_to_local(global_points, origin_x, origin_y, theta):
    centered = global_points - np.array([origin_x, origin_y])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return centered @ R.T

def create_interactive_plot(trajectory, attn_weights, save_name, gt_text, pred_text, map_feats, words):
    # Create Subplots: Left = Map, Right = Heatmap
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(f"Trajectory & Map<br><sup>GT: {gt_text} | Pred: {pred_text}</sup>", "Attention Heatmap"),
        horizontal_spacing=0.05
    )

    # --- 1. DRAW MAP ---
    # Pedestrian Crossings (Light Blue Polygons)
    for poly in map_feats['crossings']:
        # Close the loop for polygon
        x_poly = np.append(poly[:, 0], poly[0, 0])
        y_poly = np.append(poly[:, 1], poly[0, 1])
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, fill="toself", fillcolor='rgba(173, 216, 230, 0.3)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

    # Lane Dividers (Grey Dashed Lines)
    for i, line in enumerate(map_feats['dividers']):
        fig.add_trace(go.Scatter(
            x=line[:, 0], y=line[:, 1], mode='lines',
            line=dict(color='grey', width=1, dash='dash'),
            showlegend=(i==0), name='Lane Dividers', hoverinfo='skip'
        ), row=1, col=1)

    # --- 2. DRAW TRAJECTORY ---
    traj = trajectory.cpu().numpy()
    
    # Start Point
    fig.add_trace(go.Scatter(
        x=[traj[0, 0]], y=[traj[0, 1]], mode='markers',
        marker=dict(color='green', size=12, symbol='circle'),
        name='Start'
    ), row=1, col=1)
    
    # End Point
    fig.add_trace(go.Scatter(
        x=[traj[-1, 0]], y=[traj[-1, 1]], mode='markers',
        marker=dict(color='red', size=12, symbol='circle'),
        name='End'
    ), row=1, col=1)

    # Path
    fig.add_trace(go.Scatter(
        x=traj[:, 0], y=traj[:, 1], mode='lines+markers',
        line=dict(color='blue', width=4),
        marker=dict(size=4),
        name='Predicted Path'
    ), row=1, col=1)

    # Fix Aspect Ratio for Map
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)

    # --- 3. DRAW HEATMAP ---
    if len(words) > 0:
        heatmap = attn_weights.cpu().numpy().T # Transpose for [Words, Time]
        
        fig.add_trace(go.Heatmap(
            z=heatmap,
            x=np.arange(30),
            y=words,
            colorscale='Viridis',
            colorbar=dict(title='Attn'),
        ), row=1, col=2)

        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2) # Words top-to-bottom

    # --- LAYOUT ---
    fig.update_layout(
        height=700, width=1400,
        title_text="Interactive Thesis Visualization (Zoom/Pan enabled)",
        template="plotly_white",
        dragmode='pan' # Default tool is pan
    )
    
    # Save HTML
    fig.write_html(save_name)
    print(f"Saved Interactive Plot: {save_name}")

def visualize():
    with open("vocab.json") as f: vocab = json.load(f)
    print(f"Loading Model...")
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    print("Searching for samples...")
    count = 0
    for batch in loader:
        if count >= 3: break
        
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        if "right" not in gt_text and "left" not in gt_text: continue
            
        city = batch.city[0]
        origin = batch.origin[0].numpy()
        theta = batch.theta[0].item()
        
        map_feats = get_map_features(city, origin, theta)
        
        data = batch.to(model.device)
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input = data.y[0].unsqueeze(0)
            logits, attn_weights = model.captioner(global_embed, traj_input, captions=None, return_attn=True)
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            words = [model.tokenizer.idx2word[i.item()] for i in pred_ids if i.item() > 1]
            
            if len(words) > 0:
                print(f"Generating HTML for: {gt_text}")
                create_interactive_plot(
                    traj_input[0], 
                    attn_weights[0, :len(words)], 
                    f"interactive_viz_{count}.html", 
                    gt_text, pred_text, map_feats, words
                )
                count += 1

if __name__ == "__main__":
    visualize()