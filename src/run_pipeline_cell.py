# @title Step 7: Run Complete Pipeline (With Settings)
"""
Configure settings and run the complete depth-based segmentation pipeline.
This cell assumes all previous setup cells have been run.
"""

import sys
import os
import cv2
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    from torchvision import transforms
except:
    import torchvision.transforms as transforms

# ============================================================
# CREATE SETTINGS WIDGET
# ============================================================

# Create output widget for results
output = widgets.Output()

# Image settings
image_path = widgets.Text(
    value='sample_image.jpg',
    description='Image Path:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

use_existing = widgets.Checkbox(
    value=True,
    description='Use existing image if available',
    style={'description_width': 'initial'}
)

# SAM settings
run_sam = widgets.Checkbox(
    value=True,
    description='Run SAM segmentation (uncheck to use existing)',
    style={'description_width': 'initial'}
)

points_per_side = widgets.IntSlider(
    value=32, min=16, max=64, step=4,
    description='Points/side:',
    style={'description_width': 'initial'}
)

pred_iou_thresh = widgets.FloatSlider(
    value=0.88, min=0.5, max=1.0, step=0.02,
    description='IoU thresh:',
    style={'description_width': 'initial'}
)

stability_thresh = widgets.FloatSlider(
    value=0.95, min=0.5, max=1.0, step=0.02,
    description='Stability:',
    style={'description_width': 'initial'}
)

min_mask_area = widgets.IntSlider(
    value=100, min=10, max=1000, step=10,
    description='Min area:',
    style={'description_width': 'initial'}
)

# Depth settings
run_depth = widgets.Checkbox(
    value=True,
    description='Generate depth map (uncheck to use existing)',
    style={'description_width': 'initial'}
)

depth_input_size = widgets.IntSlider(
    value=518, min=256, max=1024, step=32,
    description='Depth size:',
    style={'description_width': 'initial'}
)

# Visualization settings
show_overview = widgets.Checkbox(
    value=True,
    description='Show overview visualization',
    style={'description_width': 'initial'}
)

show_detailed = widgets.Checkbox(
    value=True,
    description='Show detailed steps visualization',
    style={'description_width': 'initial'}
)

depth_colormap = widgets.Dropdown(
    options=['turbo', 'viridis', 'plasma', 'inferno', 'jet'],
    value='turbo',
    description='Colormap:',
    style={'description_width': 'initial'}
)

# Export settings
export_results = widgets.Checkbox(
    value=True,
    description='Export results to files',
    style={'description_width': 'initial'}
)

output_dir = widgets.Text(
    value='./pipeline_output',
    description='Output dir:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

# Status and run button
status_text = widgets.HTML(value='<b>Configure settings and click Run Pipeline</b>')

run_button = widgets.Button(
    description='🚀 Run Pipeline',
    button_style='success',
    tooltip='Run the complete pipeline with configured settings',
    layout=widgets.Layout(width='150px')
)

# ============================================================
# PIPELINE EXECUTION FUNCTION
# ============================================================

def run_pipeline(b):
    """Execute the pipeline with configured settings"""

    with output:
        clear_output(wait=True)

        print("="*60)
        print("RUNNING FULL PIPELINE WITH CONFIGURED SETTINGS")
        print("="*60)

        status_text.value = '<b style="color: blue;">🔄 Running pipeline...</b>'

        # Check if pipeline_results exists, if not create it
        global pipeline_results
        if 'pipeline_results' not in globals():
            pipeline_results = {}

        try:
            # ============================================================
            # STEP 1: Load and prepare image
            # ============================================================
            print("\n[1/5] Loading image...")

            global sample_img, img_rgb

            # Try to load image based on settings
            if use_existing.value and 'sample_img' in globals() and sample_img is not None:
                print(f"   Using existing loaded image")
            elif use_existing.value and 'pipeline_results' in globals() and 'image' in pipeline_results:
                sample_img = pipeline_results['image']
                print(f"   Using image from pipeline_results")
            else:
                # Load from file
                img_path = image_path.value
                if os.path.exists(img_path):
                    sample_img = cv2.imread(img_path)
                    if sample_img is not None:
                        print(f"   Loaded image from {img_path}")
                    else:
                        raise ValueError(f"Could not read {img_path}")
                else:
                    # Try default paths
                    default_paths = ["sample_image.jpg", "test_image.jpg", "input.jpg"]
                    for path in default_paths:
                        if os.path.exists(path):
                            sample_img = cv2.imread(path)
                            if sample_img is not None:
                                print(f"   Found image at {path}")
                                break

                    if sample_img is None:
                        raise ValueError(f"No image found. Please check the path: {img_path}")

            height, width = sample_img.shape[:2]
            print(f"   ✓ Image: {width}x{height}")

            # Convert to RGB
            img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

            # Store in pipeline
            pipeline_results['image'] = sample_img
            pipeline_results['image_rgb'] = img_rgb

            # ============================================================
            # STEP 2: Run SAM 2 segmentation
            # ============================================================
            if not run_sam.value and 'masks' in pipeline_results:
                masks = pipeline_results['masks']
                print(f"\n[2/5] Using existing SAM masks: {len(masks)} masks")
            else:
                print("\n[2/5] Running SAM 2 segmentation...")

                # Check if sam2 model exists
                if 'sam2' not in globals():
                    raise NameError("sam2 model not found. Please run SAM 2 initialization first.")

                # Import SAM2AutomaticMaskGenerator if needed
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

                # Create mask generator with configured settings
                mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2,
                    points_per_side=points_per_side.value,
                    pred_iou_thresh=pred_iou_thresh.value,
                    stability_score_thresh=stability_thresh.value,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=min_mask_area.value,
                )

                # Generate masks
                masks_auto = mask_generator.generate(sample_img)
                print(f"   Generated {len(masks_auto)} masks")

                # Sort by area
                masks_auto.sort(key=lambda x: x['area'], reverse=True)
                masks = [m['segmentation'] for m in masks_auto]

                # Store in pipeline
                pipeline_results['masks'] = masks
                pipeline_results['masks_auto'] = masks_auto
                print(f"   ✓ SAM segmentation complete: {len(masks)} masks")

            # ============================================================
            # STEP 3: Generate depth map
            # ============================================================
            if not run_depth.value and 'depth_map' in pipeline_results:
                depth_map = pipeline_results['depth_map']
                print(f"\n[3/5] Using existing depth map: {depth_map.shape}")
            else:
                print("\n[3/5] Generating depth map...")

                # Check if depth model exists
                if 'depth_model' not in globals():
                    raise NameError("depth_model not found. Please run DepthAnything V2 initialization first.")

                # Check device
                if 'device' not in globals():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"   Using device: {device}")

                # Prepare image for depth model
                input_size = depth_input_size.value
                img_pil = Image.fromarray(img_rgb)
                img_resized = img_pil.resize((input_size, input_size))

                # Convert to tensor
                img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

                # Generate depth
                with torch.no_grad():
                    depth_output = depth_model(img_tensor)

                # Process depth map
                depth_map = depth_output.squeeze().cpu().numpy()

                # Resize to original size if needed
                if depth_map.shape != (height, width):
                    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

                # Normalize
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

                # Store in pipeline
                pipeline_results['depth_map'] = depth_map
                print(f"   ✓ Depth map generated: {depth_map.shape}")

            # ============================================================
            # STEP 4: Create robust depth-ordered painting steps
            # ============================================================
            print("\n[4/5] Creating robust painting steps...")

            # Add parent directory to path if needed
            parent_dir = os.path.dirname(os.path.abspath('.'))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Try to import from current directory first, then parent
            try:
                from depth_ordering_grouped import create_painting_steps, visualize_painting_steps
            except ImportError:
                sys.path.insert(0, '..')
                from depth_ordering_grouped import create_painting_steps, visualize_painting_steps

            # Create painting steps with robust ordering
            painting_steps = create_painting_steps(
                masks=masks,
                depth_map=depth_map,
                img_rgb=img_rgb,
                labels=None
            )

            # Store in pipeline
            pipeline_results['painting_steps'] = painting_steps
            print(f"   ✓ Created {len(painting_steps)} painting steps")

            # ============================================================
            # STEP 5: Visualize results
            # ============================================================
            print("\n[5/5] Generating visualizations...")

            if show_overview.value:
                # Create overview visualization
                fig = plt.figure(figsize=(20, 10))

                # 1. Original image
                ax1 = plt.subplot(2, 4, 1)
                ax1.imshow(img_rgb)
                ax1.set_title('Original Image', fontsize=12, fontweight='bold')
                ax1.axis('off')

                # 2. SAM masks
                ax2 = plt.subplot(2, 4, 2)
                mask_overlay = np.zeros_like(img_rgb)
                colors_mask = plt.cm.rainbow(np.linspace(0, 1, min(20, len(masks))))
                for i, mask in enumerate(masks[:20]):
                    color = (colors_mask[i % len(colors_mask)][:3] * 255).astype(np.uint8)
                    mask_overlay[mask] = color
                ax2.imshow(mask_overlay)
                ax2.set_title(f'SAM Masks ({len(masks)} total)', fontsize=12, fontweight='bold')
                ax2.axis('off')

                # 3. Depth map
                ax3 = plt.subplot(2, 4, 3)
                im = ax3.imshow(depth_map, cmap=depth_colormap.value)
                ax3.set_title('Depth Map', fontsize=12, fontweight='bold')
                ax3.axis('off')
                plt.colorbar(im, ax=ax3, fraction=0.046)

                # 4. All painting steps
                ax4 = plt.subplot(2, 4, 4)
                step_viz = np.ones_like(img_rgb) * 255
                step_colors = plt.cm.Set3(np.linspace(0, 1, len(painting_steps)))
                for step_idx, step in enumerate(painting_steps):
                    color = (step_colors[step_idx][:3] * 255).astype(np.uint8)
                    for obj in step['objects']:
                        step_viz[obj['mask']] = color
                ax4.imshow(step_viz)
                ax4.set_title(f'Painting Steps ({len(painting_steps)} total)', fontsize=12, fontweight='bold')
                ax4.axis('off')

                # 5-8. Individual steps
                for i, step in enumerate(painting_steps[:4]):
                    ax = plt.subplot(2, 4, i + 5)

                    step_img = img_rgb.copy()
                    combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
                    for obj in step['objects']:
                        combined_mask |= obj['mask']

                    step_img[~combined_mask] = step_img[~combined_mask] * 0.3

                    ax.imshow(step_img)
                    title = f"Step {step['step']}: {step['group_name']}\n"
                    title += f"({len(step['objects'])} objects)"
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')

                plt.suptitle('Complete Pipeline Results - Robust Depth Ordering', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()

            if show_detailed.value:
                # Show detailed painting steps visualization
                print("\nDetailed painting steps visualization:")
                fig2 = visualize_painting_steps(sample_img, depth_map, painting_steps, masks)
                plt.show()

            # ============================================================
            # Print summary
            # ============================================================
            print("\n" + "="*60)
            print("PIPELINE COMPLETE!")
            print("="*60)
            print(f"\n📊 Results Summary:")
            print(f"   • Image: {width}x{height} pixels")
            print(f"   • SAM masks: {len(masks)}")
            print(f"   • Painting steps: {len(painting_steps)}")

            print(f"\n📋 Painting Steps (back to front):")
            for step in painting_steps:
                print(f"\n   Step {step['step']}: {step['group_name']}")
                print(f"      Mask indices: {step['mask_indices']}")
                print(f"      Objects: {len(step['objects'])}")

            print("\n✅ All results stored in 'pipeline_results' dictionary")

            # ============================================================
            # Export results if requested
            # ============================================================
            if export_results.value:
                print("\n💾 Exporting results...")

                # Create output directory
                os.makedirs(output_dir.value, exist_ok=True)

                # Export painting steps as JSON
                steps_data = []
                for step in painting_steps:
                    steps_data.append({
                        'step': step['step'],
                        'group_name': step['group_name'],
                        'mask_indices': step['mask_indices'],
                        'num_objects': len(step['objects'])
                    })

                json_path = os.path.join(output_dir.value, "painting_steps.json")
                with open(json_path, 'w') as f:
                    json.dump(steps_data, f, indent=2)

                # Save depth map
                np.save(os.path.join(output_dir.value, "depth_map.npy"), depth_map)

                # Save masks (compressed)
                np.savez_compressed(os.path.join(output_dir.value, "masks.npz"), masks=masks)

                print(f"   ✓ Results exported to {output_dir.value}/")

            status_text.value = '<b style="color: green;">✅ Pipeline completed successfully!</b>'

        except Exception as e:
            status_text.value = f'<b style="color: red;">❌ Error: {str(e)}</b>'
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

# Connect button handler
run_button.on_click(run_pipeline)

# ============================================================
# CREATE WIDGET LAYOUT
# ============================================================

# Group settings into sections
image_section = widgets.VBox([
    widgets.HTML('<b>📷 Image Settings</b>'),
    image_path,
    use_existing,
], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px'))

sam_section = widgets.VBox([
    widgets.HTML('<b>🎯 SAM Settings</b>'),
    run_sam,
    points_per_side,
    pred_iou_thresh,
    stability_thresh,
    min_mask_area,
], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px'))

depth_section = widgets.VBox([
    widgets.HTML('<b>🏔️ Depth Settings</b>'),
    run_depth,
    depth_input_size,
], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px'))

viz_section = widgets.VBox([
    widgets.HTML('<b>📊 Visualization Settings</b>'),
    show_overview,
    show_detailed,
    depth_colormap,
], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px'))

export_section = widgets.VBox([
    widgets.HTML('<b>💾 Export Settings</b>'),
    export_results,
    output_dir,
], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px'))

# Create two columns
left_column = widgets.VBox([image_section, sam_section])
right_column = widgets.VBox([depth_section, viz_section, export_section])

# Main layout
settings_layout = widgets.HBox([left_column, right_column])

main_widget = widgets.VBox([
    widgets.HTML('<h3>🎛️ Pipeline Settings</h3>'),
    settings_layout,
    widgets.HTML('<hr>'),
    widgets.HBox([run_button, status_text]),
    output
], layout=widgets.Layout(border='2px solid #ccc', padding='15px', border_radius='10px'))

# Display the widget
display(main_widget)

print("🎛️ Pipeline Settings Widget Ready!")
print("Configure settings above, then click 'Run Pipeline' to execute.")