# @title Step 7: Complete Pipeline with Settings Widget
"""
Interactive widget to configure and run the complete pipeline with all settings.
Configure all parameters before running the entire pipeline.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import sys
try:
    from torchvision import transforms
except:
    import torchvision.transforms as transforms

# Create output widget for results
output = widgets.Output()

# ============================================================
# IMAGE SETTINGS
# ============================================================
image_settings_label = widgets.HTML('<h3>📷 Image Settings</h3>')

image_path_text = widgets.Text(
    value='sample_image.jpg',
    placeholder='Path to image',
    description='Image Path:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

use_existing_image = widgets.Checkbox(
    value=True,
    description='Use existing loaded image if available',
    style={'description_width': 'initial'}
)

# ============================================================
# SAM 2 SEGMENTATION SETTINGS
# ============================================================
sam_settings_label = widgets.HTML('<h3>🎯 SAM 2 Segmentation Settings</h3>')

run_sam_checkbox = widgets.Checkbox(
    value=True,
    description='Run SAM segmentation',
    style={'description_width': 'initial'}
)

points_per_side = widgets.IntSlider(
    value=32,
    min=16,
    max=64,
    step=4,
    description='Points per side:',
    style={'description_width': 'initial'},
    tooltip='Number of points to sample per side for mask generation'
)

pred_iou_thresh = widgets.FloatSlider(
    value=0.88,
    min=0.5,
    max=1.0,
    step=0.02,
    description='Pred IoU threshold:',
    style={'description_width': 'initial'},
    tooltip='Filter masks by predicted IoU quality'
)

stability_score_thresh = widgets.FloatSlider(
    value=0.95,
    min=0.5,
    max=1.0,
    step=0.02,
    description='Stability threshold:',
    style={'description_width': 'initial'},
    tooltip='Filter masks by stability score'
)

min_mask_region = widgets.IntSlider(
    value=100,
    min=10,
    max=1000,
    step=10,
    description='Min mask area:',
    style={'description_width': 'initial'},
    tooltip='Minimum number of pixels for a valid mask'
)

crop_n_layers = widgets.IntSlider(
    value=1,
    min=0,
    max=3,
    step=1,
    description='Crop layers:',
    style={'description_width': 'initial'},
    tooltip='Number of crop layers for hierarchical mask generation'
)

# ============================================================
# DEPTH ESTIMATION SETTINGS
# ============================================================
depth_settings_label = widgets.HTML('<h3>🏔️ Depth Estimation Settings</h3>')

run_depth_checkbox = widgets.Checkbox(
    value=True,
    description='Generate depth map',
    style={'description_width': 'initial'}
)

depth_model_size = widgets.Dropdown(
    options=['vits', 'vitb', 'vitl'],
    value='vitb',
    description='Model size:',
    style={'description_width': 'initial'},
    tooltip='ViT model size (s=small, b=base, l=large)'
)

depth_input_size = widgets.IntSlider(
    value=518,
    min=256,
    max=1024,
    step=32,
    description='Input size:',
    style={'description_width': 'initial'},
    tooltip='Size to resize image for depth estimation'
)

# ============================================================
# DEPTH ORDERING SETTINGS
# ============================================================
ordering_settings_label = widgets.HTML('<h3>🎨 Depth Ordering & Grouping Settings</h3>')

use_robust_ordering = widgets.Checkbox(
    value=True,
    description='Use robust depth ordering',
    style={'description_width': 'initial'},
    tooltip='Use improved depth ordering with occlusion handling'
)

erosion_iterations = widgets.IntSlider(
    value=2,
    min=0,
    max=5,
    step=1,
    description='Erosion iterations:',
    style={'description_width': 'initial'},
    tooltip='Erosion iterations for robust depth score'
)

group_similar = widgets.Checkbox(
    value=True,
    description='Group similar objects',
    style={'description_width': 'initial'},
    tooltip='Group objects with similar properties in same step'
)

max_depth_bands = widgets.IntSlider(
    value=3,
    min=2,
    max=5,
    step=1,
    description='Depth bands:',
    style={'description_width': 'initial'},
    tooltip='Number of depth bands for grouping (far/mid/near)'
)

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
viz_settings_label = widgets.HTML('<h3>📊 Visualization Settings</h3>')

show_overview = widgets.Checkbox(
    value=True,
    description='Show overview visualization',
    style={'description_width': 'initial'}
)

show_detailed_steps = widgets.Checkbox(
    value=True,
    description='Show detailed step visualization',
    style={'description_width': 'initial'}
)

colormap_dropdown = widgets.Dropdown(
    options=['turbo', 'viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow'],
    value='turbo',
    description='Depth colormap:',
    style={'description_width': 'initial'}
)

figure_size = widgets.IntSlider(
    value=20,
    min=10,
    max=30,
    step=2,
    description='Figure width:',
    style={'description_width': 'initial'},
    tooltip='Width of visualization figures'
)

# ============================================================
# EXPORT SETTINGS
# ============================================================
export_settings_label = widgets.HTML('<h3>💾 Export Settings</h3>')

export_results = widgets.Checkbox(
    value=True,
    description='Export results to files',
    style={'description_width': 'initial'}
)

output_dir_text = widgets.Text(
    value='./pipeline_output',
    placeholder='Output directory',
    description='Output dir:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

export_format = widgets.SelectMultiple(
    options=['JSON (steps)', 'NPY (depth)', 'NPZ (masks)', 'PNG (visualizations)'],
    value=['JSON (steps)', 'PNG (visualizations)'],
    description='Export formats:',
    style={'description_width': 'initial'},
    rows=4
)

# ============================================================
# STATUS AND RUN CONTROLS
# ============================================================
status_text = widgets.HTML(value='<b>Ready to configure and run pipeline</b>')

run_button = widgets.Button(
    description='🚀 Run Complete Pipeline',
    button_style='success',
    tooltip='Run the complete pipeline with configured settings',
    layout=widgets.Layout(width='200px', height='40px')
)

progress_bar = widgets.IntProgress(
    value=0,
    min=0,
    max=5,
    description='Progress:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

# ============================================================
# PIPELINE EXECUTION FUNCTION
# ============================================================
def run_complete_pipeline(b):
    """Execute the complete pipeline with configured settings"""

    with output:
        clear_output(wait=True)

        # Reset progress
        progress_bar.value = 0
        status_text.value = '<b style="color: blue;">🔄 Starting pipeline...</b>'

        # Check if pipeline_results exists
        global pipeline_results
        if 'pipeline_results' not in globals():
            pipeline_results = {}

        try:
            # ============================================================
            # STEP 1: Load Image
            # ============================================================
            print("="*60)
            print("RUNNING COMPLETE PIPELINE WITH CUSTOM SETTINGS")
            print("="*60)
            print("\n[1/5] Loading image...")

            global sample_img, img_rgb

            # Try to load image based on settings
            if use_existing_image.value and 'sample_img' in globals() and sample_img is not None:
                print(f"   Using existing loaded image")
            else:
                image_path = image_path_text.value
                if os.path.exists(image_path):
                    sample_img = cv2.imread(image_path)
                    if sample_img is not None:
                        print(f"   Loaded image from {image_path}")
                    else:
                        raise ValueError(f"Could not read image from {image_path}")
                else:
                    raise FileNotFoundError(f"Image not found: {image_path}")

            height, width = sample_img.shape[:2]
            img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

            pipeline_results['image'] = sample_img
            pipeline_results['image_rgb'] = img_rgb

            print(f"   ✓ Image ready: {width}x{height}")
            progress_bar.value = 1

            # ============================================================
            # STEP 2: SAM Segmentation
            # ============================================================
            if run_sam_checkbox.value:
                print("\n[2/5] Running SAM 2 segmentation...")

                # Check for existing masks
                if 'masks' in pipeline_results and not run_sam_checkbox.value:
                    masks = pipeline_results['masks']
                    print(f"   Using existing masks: {len(masks)}")
                else:
                    # Import if needed
                    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

                    # Create mask generator with custom settings
                    mask_generator = SAM2AutomaticMaskGenerator(
                        model=sam2,
                        points_per_side=points_per_side.value,
                        pred_iou_thresh=pred_iou_thresh.value,
                        stability_score_thresh=stability_score_thresh.value,
                        crop_n_layers=crop_n_layers.value,
                        crop_n_points_downscale_factor=2,
                        min_mask_region_area=min_mask_region.value,
                    )

                    # Generate masks
                    masks_auto = mask_generator.generate(sample_img)
                    masks_auto.sort(key=lambda x: x['area'], reverse=True)
                    masks = [m['segmentation'] for m in masks_auto]

                    pipeline_results['masks'] = masks
                    pipeline_results['masks_auto'] = masks_auto

                    print(f"   ✓ Generated {len(masks)} masks")
            else:
                if 'masks' in pipeline_results:
                    masks = pipeline_results['masks']
                    print("\n[2/5] Using existing masks")
                else:
                    raise ValueError("No masks available and SAM is disabled")

            progress_bar.value = 2

            # ============================================================
            # STEP 3: Depth Estimation
            # ============================================================
            if run_depth_checkbox.value:
                print("\n[3/5] Generating depth map...")

                # Check device
                if 'device' not in globals():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Prepare image
                input_size = depth_input_size.value
                img_pil = Image.fromarray(img_rgb)
                img_resized = img_pil.resize((input_size, input_size))
                img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

                # Generate depth
                with torch.no_grad():
                    depth_output = depth_model(img_tensor)

                depth_map = depth_output.squeeze().cpu().numpy()

                # Resize to original
                if depth_map.shape != (height, width):
                    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

                # Normalize
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

                pipeline_results['depth_map'] = depth_map
                print(f"   ✓ Depth map generated: {depth_map.shape}")
            else:
                if 'depth_map' in pipeline_results:
                    depth_map = pipeline_results['depth_map']
                    print("\n[3/5] Using existing depth map")
                else:
                    raise ValueError("No depth map available and depth estimation is disabled")

            progress_bar.value = 3

            # ============================================================
            # STEP 4: Depth Ordering
            # ============================================================
            print("\n[4/5] Creating depth-ordered painting steps...")

            # Import depth ordering functions
            sys.path.insert(0, '..')
            from depth_ordering_grouped import create_painting_steps, visualize_painting_steps

            # Create painting steps with settings
            if use_robust_ordering.value:
                # Use the robust ordering from our updated module
                painting_steps = create_painting_steps(
                    masks=masks,
                    depth_map=depth_map,
                    img_rgb=img_rgb,
                    labels=None
                )
            else:
                # Simple depth-based ordering
                painting_steps = []
                for i, mask in enumerate(masks):
                    masked_depths = depth_map[mask]
                    if len(masked_depths) > 0:
                        painting_steps.append({
                            'step': i + 1,
                            'group_name': f'object_{i}',
                            'mask_indices': [i],
                            'objects': [{
                                'id': i,
                                'mask': mask,
                                'depth_score': np.median(masked_depths),
                                'area_ratio': np.sum(mask) / (height * width)
                            }]
                        })

            pipeline_results['painting_steps'] = painting_steps
            print(f"   ✓ Created {len(painting_steps)} painting steps")
            progress_bar.value = 4

            # ============================================================
            # STEP 5: Visualization
            # ============================================================
            print("\n[5/5] Generating visualizations...")

            if show_overview.value:
                # Create overview visualization
                fig = plt.figure(figsize=(figure_size.value, figure_size.value // 2))

                # Original image
                ax1 = plt.subplot(2, 4, 1)
                ax1.imshow(img_rgb)
                ax1.set_title('Original Image', fontsize=12, fontweight='bold')
                ax1.axis('off')

                # SAM masks
                ax2 = plt.subplot(2, 4, 2)
                mask_overlay = np.zeros_like(img_rgb)
                colors = plt.cm.rainbow(np.linspace(0, 1, min(20, len(masks))))
                for i, mask in enumerate(masks[:20]):
                    color = (colors[i][:3] * 255).astype(np.uint8)
                    mask_overlay[mask] = color
                ax2.imshow(mask_overlay)
                ax2.set_title(f'SAM Masks ({len(masks)})', fontsize=12, fontweight='bold')
                ax2.axis('off')

                # Depth map
                ax3 = plt.subplot(2, 4, 3)
                im = ax3.imshow(depth_map, cmap=colormap_dropdown.value)
                ax3.set_title('Depth Map', fontsize=12, fontweight='bold')
                ax3.axis('off')
                plt.colorbar(im, ax=ax3, fraction=0.046)

                # Painting steps
                ax4 = plt.subplot(2, 4, 4)
                step_viz = np.ones_like(img_rgb) * 255
                step_colors = plt.cm.Set3(np.linspace(0, 1, len(painting_steps)))
                for idx, step in enumerate(painting_steps):
                    color = (step_colors[idx][:3] * 255).astype(np.uint8)
                    for obj in step['objects']:
                        step_viz[obj['mask']] = color
                ax4.imshow(step_viz)
                ax4.set_title(f'Steps ({len(painting_steps)})', fontsize=12, fontweight='bold')
                ax4.axis('off')

                # Individual steps preview
                for i, step in enumerate(painting_steps[:4]):
                    ax = plt.subplot(2, 4, i + 5)
                    step_img = img_rgb.copy()
                    combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
                    for obj in step['objects']:
                        combined_mask |= obj['mask']
                    step_img[~combined_mask] = step_img[~combined_mask] * 0.3
                    ax.imshow(step_img)
                    ax.set_title(f"Step {step['step']}", fontsize=10)
                    ax.axis('off')

                plt.suptitle('Pipeline Results Overview', fontsize=14, fontweight='bold')
                plt.tight_layout()

                if export_results.value and 'PNG (visualizations)' in export_format.value:
                    os.makedirs(output_dir_text.value, exist_ok=True)
                    plt.savefig(os.path.join(output_dir_text.value, 'pipeline_overview.png'),
                               dpi=150, bbox_inches='tight')

                plt.show()

            if show_detailed_steps.value:
                # Show detailed steps visualization
                fig2 = visualize_painting_steps(sample_img, depth_map, painting_steps, masks)

                if export_results.value and 'PNG (visualizations)' in export_format.value:
                    fig2.savefig(os.path.join(output_dir_text.value, 'painting_steps_detailed.png'),
                                dpi=150, bbox_inches='tight')

                plt.show()

            progress_bar.value = 5

            # ============================================================
            # Export Results
            # ============================================================
            if export_results.value:
                print("\n💾 Exporting results...")
                os.makedirs(output_dir_text.value, exist_ok=True)

                if 'JSON (steps)' in export_format.value:
                    steps_data = [{
                        'step': step['step'],
                        'group_name': step['group_name'],
                        'mask_indices': step['mask_indices'],
                        'num_objects': len(step['objects'])
                    } for step in painting_steps]

                    with open(os.path.join(output_dir_text.value, 'painting_steps.json'), 'w') as f:
                        json.dump(steps_data, f, indent=2)
                    print(f"   ✓ Exported painting steps JSON")

                if 'NPY (depth)' in export_format.value:
                    np.save(os.path.join(output_dir_text.value, 'depth_map.npy'), depth_map)
                    print(f"   ✓ Exported depth map")

                if 'NPZ (masks)' in export_format.value:
                    np.savez_compressed(os.path.join(output_dir_text.value, 'masks.npz'), masks=masks)
                    print(f"   ✓ Exported masks")

            # ============================================================
            # Summary
            # ============================================================
            print("\n" + "="*60)
            print("PIPELINE COMPLETE!")
            print("="*60)
            print(f"\n📊 Results Summary:")
            print(f"   • Image: {width}x{height} pixels")
            print(f"   • SAM masks: {len(masks)}")
            print(f"   • Painting steps: {len(painting_steps)}")
            print(f"\n📋 Painting Steps:")

            for step in painting_steps[:5]:  # Show first 5 steps
                print(f"   Step {step['step']}: {step['group_name']} ({len(step['objects'])} objects)")

            if len(painting_steps) > 5:
                print(f"   ... and {len(painting_steps) - 5} more steps")

            status_text.value = '<b style="color: green;">✅ Pipeline completed successfully!</b>'

        except Exception as e:
            status_text.value = f'<b style="color: red;">❌ Error: {str(e)}</b>'
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            progress_bar.value = 0

# Connect button handler
run_button.on_click(run_complete_pipeline)

# ============================================================
# CREATE TABBED INTERFACE
# ============================================================
tab = widgets.Tab()

# Image & Input tab
image_tab = widgets.VBox([
    image_settings_label,
    image_path_text,
    use_existing_image,
    widgets.HTML('<hr>'),
])

# SAM Settings tab
sam_tab = widgets.VBox([
    sam_settings_label,
    run_sam_checkbox,
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    min_mask_region,
    crop_n_layers,
])

# Depth Settings tab
depth_tab = widgets.VBox([
    depth_settings_label,
    run_depth_checkbox,
    depth_model_size,
    depth_input_size,
])

# Ordering Settings tab
ordering_tab = widgets.VBox([
    ordering_settings_label,
    use_robust_ordering,
    erosion_iterations,
    group_similar,
    max_depth_bands,
])

# Visualization tab
viz_tab = widgets.VBox([
    viz_settings_label,
    show_overview,
    show_detailed_steps,
    colormap_dropdown,
    figure_size,
])

# Export tab
export_tab = widgets.VBox([
    export_settings_label,
    export_results,
    output_dir_text,
    export_format,
])

# Set tab contents
tab.children = [image_tab, sam_tab, depth_tab, ordering_tab, viz_tab, export_tab]
tab.titles = ['📷 Image', '🎯 SAM', '🏔️ Depth', '🎨 Ordering', '📊 Viz', '💾 Export']

# Main layout
main_layout = widgets.VBox([
    widgets.HTML('<h2>🎛️ Complete Pipeline Settings</h2>'),
    widgets.HTML('<p>Configure all settings below, then click Run to execute the complete pipeline.</p>'),
    tab,
    widgets.HTML('<hr>'),
    widgets.HBox([run_button, progress_bar]),
    status_text,
    output
], layout=widgets.Layout(padding='10px', border='2px solid #ccc', border_radius='10px'))

# Display the widget
display(main_layout)

print("🎛️ Pipeline Settings Widget Ready!")
print("="*60)
print("Configure your settings in the tabs above, then click 'Run Complete Pipeline'")
print("All parameters can be adjusted before running the pipeline.")