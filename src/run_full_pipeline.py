#!/usr/bin/env python3
"""
Standalone pipeline runner for depth-based object segmentation and ordering.
Run this file or paste its content into a notebook cell to execute the full pipeline.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import json
from pathlib import Path

# Import SAM 2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import Depth Anything V2
from depth_anything_v2.dpt import DepthAnythingV2

# Import our custom depth ordering function
from depth_ordering_grouped import create_painting_steps, visualize_painting_steps


def run_full_pipeline(
    image_path,
    sam_checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt",
    sam_config="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    depth_model_path="./sam2/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
    output_dir="./pipeline_output",
    visualize=True,
    export_results=True
):
    """
    Run the complete depth-based segmentation and ordering pipeline.

    Args:
        image_path: Path to input image
        sam_checkpoint: Path to SAM 2 checkpoint
        sam_config: Path to SAM 2 config file
        depth_model_path: Path to Depth Anything V2 model
        output_dir: Directory for output files
        visualize: Whether to show visualizations
        export_results: Whether to export results to files

    Returns:
        Dictionary containing all pipeline results
    """

    print("="*60)
    print("RUNNING FULL DEPTH-BASED SEGMENTATION PIPELINE")
    print("="*60)

    # Create output directory if needed
    if export_results:
        os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # STEP 1: Load Image
    # ============================================================
    print("\n[1/5] Loading image...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_bgr.shape[:2]
    print(f"   ✓ Image loaded: {width}x{height}")

    # ============================================================
    # STEP 2: Initialize SAM 2
    # ============================================================
    print("\n[2/5] Initializing SAM 2...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Build SAM 2 model
    sam2_model = build_sam2(sam_config, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Set image
    predictor.set_image(img_bgr)
    print("   ✓ SAM 2 initialized and image set")

    # ============================================================
    # STEP 3: Generate Masks with SAM 2
    # ============================================================
    print("\n[3/5] Generating segmentation masks...")

    # Generate automatic masks
    masks_generator = predictor.generate(
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=False,
        return_logits=False
    )

    # For automatic mask generation, we need to use automatic mask generator
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # Generate masks
    masks_auto = mask_generator.generate(img_bgr)

    print(f"   ✓ Generated {len(masks_auto)} masks")

    # Sort masks by area (largest first)
    masks_auto.sort(key=lambda x: x['area'], reverse=True)

    # Extract just the segmentation arrays
    masks = [m['segmentation'] for m in masks_auto]

    # ============================================================
    # STEP 4: Generate Depth Map
    # ============================================================
    print("\n[4/5] Generating depth map with Depth Anything V2...")

    # Initialize Depth Anything V2
    depth_model = DepthAnythingV2(
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384]
    )

    # Load weights
    depth_model.load_state_dict(torch.load(depth_model_path, map_location=device))
    depth_model = depth_model.to(device).eval()

    # Prepare image for depth estimation
    img_pil = Image.fromarray(img_rgb)

    # Resize to model input size
    depth_input = img_pil.resize((518, 518))
    depth_input = np.array(depth_input) / 255.0
    depth_input = torch.from_numpy(depth_input).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Generate depth map
    with torch.no_grad():
        depth_output = depth_model(depth_input)

    # Process depth map
    depth_map = depth_output.squeeze().cpu().numpy()

    # Resize back to original size
    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    print(f"   ✓ Depth map generated: {depth_map.shape}")

    # ============================================================
    # STEP 5: Create Robust Depth-Ordered Painting Steps
    # ============================================================
    print("\n[5/5] Creating depth-ordered painting steps...")

    painting_steps = create_painting_steps(
        masks=masks,
        depth_map=depth_map,
        img_rgb=img_rgb,
        labels=None  # Add labels if you have them
    )

    print(f"   ✓ Created {len(painting_steps)} painting steps")

    # ============================================================
    # Visualize Results
    # ============================================================
    if visualize:
        print("\n📊 Generating visualizations...")

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))

        # 1. Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. All masks overlay
        ax2 = plt.subplot(2, 3, 2)
        mask_overlay = np.zeros_like(img_rgb)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
        for i, mask in enumerate(masks[:20]):  # Show first 20 masks
            color = (colors[i][:3] * 255).astype(np.uint8)
            mask_overlay[mask] = color
        ax2.imshow(mask_overlay)
        ax2.set_title(f'All Masks ({len(masks)} total)', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. Depth map
        ax3 = plt.subplot(2, 3, 3)
        im = ax3.imshow(depth_map, cmap='turbo')
        ax3.set_title('Depth Map', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # 4-6. First three painting steps
        for i, step in enumerate(painting_steps[:3]):
            ax = plt.subplot(2, 3, i + 4)

            # Create step visualization
            step_img = img_rgb.copy().astype(np.float32)
            overlay = np.zeros_like(step_img)

            # Combine all masks in this step
            combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
            for obj in step['objects']:
                combined_mask |= obj['mask']

            # Apply colored overlay
            step_color = plt.cm.Set3(i / max(3, len(painting_steps)))[:3]
            for c in range(3):
                overlay[:, :, c][combined_mask] = step_color[c] * 255

            # Blend with original
            step_img[combined_mask] = step_img[combined_mask] * 0.4 + overlay[combined_mask] * 0.6

            # Draw boundaries
            from skimage import segmentation
            for obj in step['objects']:
                boundaries = segmentation.find_boundaries(obj['mask'], mode='outer')
                step_img[boundaries] = [255, 255, 255]

            ax.imshow(step_img.astype(np.uint8))
            ax.set_title(f"Step {step['step']}: {step['group_name']}\n({len(step['objects'])} objects)",
                        fontsize=10)
            ax.axis('off')

        plt.suptitle('Depth-Based Segmentation Pipeline Results', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if export_results:
            vis_path = os.path.join(output_dir, 'pipeline_visualization.png')
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved visualization to {vis_path}")

        if visualize:
            plt.show()

        # Also create the detailed painting steps visualization
        fig2 = visualize_painting_steps(img_bgr, depth_map, painting_steps, masks)

        if export_results:
            steps_vis_path = os.path.join(output_dir, 'painting_steps_detailed.png')
            fig2.savefig(steps_vis_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved detailed steps to {steps_vis_path}")

        if visualize:
            plt.show()

    # ============================================================
    # Export Results
    # ============================================================
    if export_results:
        print("\n💾 Exporting results...")

        # Export painting steps as JSON
        steps_data = []
        for step in painting_steps:
            steps_data.append({
                'step': step['step'],
                'group_name': step['group_name'],
                'mask_indices': step['mask_indices'],
                'num_objects': len(step['objects'])
            })

        json_path = os.path.join(output_dir, 'painting_steps.json')
        with open(json_path, 'w') as f:
            json.dump(steps_data, f, indent=2)
        print(f"   ✓ Saved steps data to {json_path}")

        # Export depth map
        depth_path = os.path.join(output_dir, 'depth_map.npy')
        np.save(depth_path, depth_map)
        print(f"   ✓ Saved depth map to {depth_path}")

        # Export masks
        masks_path = os.path.join(output_dir, 'masks.npz')
        np.savez_compressed(masks_path, masks=masks)
        print(f"   ✓ Saved masks to {masks_path}")

    # ============================================================
    # Print Summary
    # ============================================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📊 Results Summary:")
    print(f"   • Input image: {width}x{height} pixels")
    print(f"   • Masks generated: {len(masks)}")
    print(f"   • Painting steps: {len(painting_steps)}")
    print(f"\n📋 Painting Steps Order:")

    for step in painting_steps:
        print(f"\n   Step {step['step']}: {step['group_name']}")
        print(f"      - Masks: {step['mask_indices']}")
        print(f"      - Objects: {len(step['objects'])}")
        if step['objects']:
            areas = [obj['area_ratio'] * 100 for obj in step['objects']]
            print(f"      - Coverage: {sum(areas):.1f}% of image")

    # Return all results
    results = {
        'image': img_rgb,
        'masks': masks,
        'depth_map': depth_map,
        'painting_steps': painting_steps,
        'image_size': (width, height),
        'device': str(device)
    }

    return results


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Configure pipeline parameters
    IMAGE_PATH = "sample_image.jpg"  # Change this to your image path

    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        print("Please update IMAGE_PATH in the script to point to your image.")
        exit(1)

    # Run the pipeline
    try:
        results = run_full_pipeline(
            image_path=IMAGE_PATH,
            sam_checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt",
            sam_config="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            depth_model_path="./sam2/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
            output_dir="./pipeline_output",
            visualize=True,
            export_results=True
        )

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# NOTEBOOK CELL VERSION
# ============================================================
"""
To use in a Jupyter notebook, paste this into a cell:

# Set your image path
IMAGE_PATH = "your_image.jpg"

# Run the full pipeline
results = run_full_pipeline(
    image_path=IMAGE_PATH,
    sam_checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt",
    sam_config="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    depth_model_path="./sam2/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
    output_dir="./pipeline_output",
    visualize=True,
    export_results=True
)

# Access results
print(f"Generated {len(results['masks'])} masks")
print(f"Created {len(results['painting_steps'])} painting steps")

# The results dictionary contains:
# - results['image']: RGB image array
# - results['masks']: List of segmentation masks
# - results['depth_map']: Normalized depth map
# - results['painting_steps']: Ordered painting steps
# - results['image_size']: (width, height) tuple
"""