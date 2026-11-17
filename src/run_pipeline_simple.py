"""
Simple pipeline runner - paste this into a notebook cell after running import/setup cells.
Assumes SAM2, DepthAnything V2, and other dependencies are already imported and initialized.
"""

# Run the complete pipeline without widgets
print("="*60)
print("RUNNING FULL PIPELINE (NO WIDGETS)")
print("="*60)

# ============================================================
# STEP 1: Load and prepare image
# ============================================================
print("\n[1/5] Loading image...")

# Load image (update path as needed)
image_path = "sample_image.jpg"  # UPDATE THIS PATH
sample_img = cv2.imread(image_path)
if sample_img is None:
    raise ValueError(f"Could not load image from {image_path}")

height, width = sample_img.shape[:2]
print(f"✓ Image loaded: {width}x{height}")

# Convert to RGB for visualization
img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

# ============================================================
# STEP 2: Run SAM 2 segmentation
# ============================================================
print("\n[2/5] Running SAM 2 segmentation...")

# Set image in predictor
predictor.set_image(sample_img)

# Generate automatic masks
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
masks_auto = mask_generator.generate(sample_img)
print(f"✓ Generated {len(masks_auto)} masks")

# Sort by area and extract segmentations
masks_auto.sort(key=lambda x: x['area'], reverse=True)
masks = [m['segmentation'] for m in masks_auto]

# ============================================================
# STEP 3: Generate depth map
# ============================================================
print("\n[3/5] Generating depth map...")

# Prepare image for depth model
input_size = 518
img_pil = Image.fromarray(img_rgb)
img_resized = img_pil.resize((input_size, input_size))

# Convert to tensor
img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

# Generate depth
with torch.no_grad():
    depth_output = depth_model(img_tensor)

# Process depth map
depth_map = depth_output.squeeze().cpu().numpy()
depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

# Normalize
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
print(f"✓ Depth map generated: {depth_map.shape}")

# ============================================================
# STEP 4: Create robust depth-ordered painting steps
# ============================================================
print("\n[4/5] Creating painting steps...")

# Import the robust ordering function
from depth_ordering_grouped import create_painting_steps, visualize_painting_steps

# Create painting steps
painting_steps = create_painting_steps(
    masks=masks,
    depth_map=depth_map,
    img_rgb=img_rgb,
    labels=None
)

print(f"✓ Created {len(painting_steps)} painting steps")

# ============================================================
# STEP 5: Visualize results
# ============================================================
print("\n[5/5] Generating visualizations...")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 10))

# 1. Original image
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=12, fontweight='bold')
ax1.axis('off')

# 2. SAM masks
ax2 = plt.subplot(2, 4, 2)
mask_overlay = np.zeros_like(img_rgb)
colors = plt.cm.rainbow(np.linspace(0, 1, min(20, len(masks))))
for i, mask in enumerate(masks[:20]):
    color = (colors[i % len(colors)][:3] * 255).astype(np.uint8)
    mask_overlay[mask] = color
ax2.imshow(mask_overlay)
ax2.set_title(f'SAM Masks ({len(masks)} total)', fontsize=12, fontweight='bold')
ax2.axis('off')

# 3. Depth map
ax3 = plt.subplot(2, 4, 3)
im = ax3.imshow(depth_map, cmap='turbo')
ax3.set_title('Depth Map', fontsize=12, fontweight='bold')
ax3.axis('off')
plt.colorbar(im, ax=ax3, fraction=0.046)

# 4. All steps combined
ax4 = plt.subplot(2, 4, 4)
step_viz = np.ones_like(img_rgb) * 255
step_colors = plt.cm.Set3(np.linspace(0, 1, len(painting_steps)))
for step_idx, step in enumerate(painting_steps):
    color = (step_colors[step_idx][:3] * 255).astype(np.uint8)
    for obj in step['objects']:
        step_viz[obj['mask']] = color
ax4.imshow(step_viz)
ax4.set_title(f'All Steps ({len(painting_steps)} total)', fontsize=12, fontweight='bold')
ax4.axis('off')

# 5-8. Individual steps
for i, step in enumerate(painting_steps[:4]):
    ax = plt.subplot(2, 4, i + 5)

    # Create step visualization
    step_img = img_rgb.copy()
    combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    for obj in step['objects']:
        combined_mask |= obj['mask']

    # Highlight this step
    step_img[~combined_mask] = step_img[~combined_mask] * 0.3

    ax.imshow(step_img)
    ax.set_title(f"Step {step['step']}: {step['group_name']}\n({len(step['objects'])} objects)",
                fontsize=10)
    ax.axis('off')

plt.suptitle('Complete Pipeline Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Also show detailed painting steps visualization
fig2 = visualize_painting_steps(sample_img, depth_map, painting_steps, masks)
plt.show()

# ============================================================
# Store results in pipeline_results dictionary
# ============================================================
pipeline_results = {
    'image': sample_img,
    'image_rgb': img_rgb,
    'masks': masks,
    'masks_auto': masks_auto,
    'depth_map': depth_map,
    'painting_steps': painting_steps,
    'image_size': (width, height)
}

# ============================================================
# Print summary
# ============================================================
print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print(f"\n📊 Results Summary:")
print(f"   • Image: {width}x{height} pixels")
print(f"   • Masks: {len(masks)}")
print(f"   • Steps: {len(painting_steps)}")

print(f"\n📋 Painting Steps Order:")
for step in painting_steps:
    print(f"\n   Step {step['step']}: {step['group_name']}")
    print(f"      Masks: {step['mask_indices']}")
    print(f"      Objects: {len(step['objects'])}")

print("\n✅ Results stored in 'pipeline_results' dictionary")
print("   Access with: pipeline_results['masks'], pipeline_results['depth_map'], etc.")

# ============================================================
# Optional: Export results
# ============================================================
export_results = True  # Set to True to export

if export_results:
    import json

    print("\n💾 Exporting results...")

    # Create output directory
    output_dir = "./pipeline_output"
    os.makedirs(output_dir, exist_ok=True)

    # Export painting steps
    steps_data = [{
        'step': step['step'],
        'group_name': step['group_name'],
        'mask_indices': step['mask_indices'],
        'num_objects': len(step['objects'])
    } for step in painting_steps]

    with open(f"{output_dir}/painting_steps.json", 'w') as f:
        json.dump(steps_data, f, indent=2)

    # Save visualizations
    fig.savefig(f"{output_dir}/pipeline_overview.png", dpi=150, bbox_inches='tight')
    fig2.savefig(f"{output_dir}/painting_steps.png", dpi=150, bbox_inches='tight')

    # Save depth map
    np.save(f"{output_dir}/depth_map.npy", depth_map)

    print(f"   ✓ Results exported to {output_dir}/")