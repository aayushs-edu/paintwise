# @title Step 6: Robust Depth-Based Object Grouping for Painting Steps
"""
Robust depth ordering and grouping of SAM-detected objects.
Orders objects from background to foreground and groups them into painting steps.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_robust_depth_score(mask, depth_map, erosion_iterations=2):
    """
    Calculate robust depth score for a mask using median of eroded mask.

    Args:
        mask: Binary mask
        depth_map: Depth map
        erosion_iterations: Number of erosion iterations to remove edge pixels

    Returns:
        Robust depth score (median of eroded mask)
    """
    # Erode mask to focus on center pixels
    if erosion_iterations > 0 and np.sum(mask) > 100:
        eroded = ndimage.binary_erosion(mask, iterations=erosion_iterations)
        if np.sum(eroded) > 10:  # Use eroded if it has enough pixels
            mask = eroded

    masked_depths = depth_map[mask]
    if len(masked_depths) == 0:
        return 1.0  # Default to far if no pixels

    return np.median(masked_depths)


def check_occlusion(mask1, mask2, depth_map):
    """
    Check if mask1 occludes mask2 by comparing boundary depths.

    Returns:
        True if mask1 is in front of mask2 at their boundary
    """
    # Find boundary pixels where masks touch
    dilated1 = ndimage.binary_dilation(mask1, iterations=1)
    dilated2 = ndimage.binary_dilation(mask2, iterations=1)
    boundary = dilated1 & dilated2

    if not np.any(boundary):
        return False  # Masks don't touch

    # Compare depths at boundary
    boundary_depths1 = depth_map[mask1 & boundary] if np.any(mask1 & boundary) else []
    boundary_depths2 = depth_map[mask2 & boundary] if np.any(mask2 & boundary) else []

    if len(boundary_depths1) > 0 and len(boundary_depths2) > 0:
        # Higher depth value means closer to camera
        return np.median(boundary_depths1) > np.median(boundary_depths2)

    return False


def create_painting_steps(masks, depth_map, img_rgb=None, labels=None):
    """
    Create painting steps with robust depth ordering and grouping.

    Args:
        masks: List of SAM masks
        depth_map: Normalized depth map from DepthAnything V2
        img_rgb: RGB image for color similarity (optional)
        labels: Object labels if available (optional)

    Returns:
        List of painting steps, each containing grouped objects
    """

    # Extract all objects with properties
    all_objects = []
    image_area = depth_map.shape[0] * depth_map.shape[1]

    for i, mask in enumerate(masks):
        if isinstance(mask, dict):
            mask_array = mask['segmentation']
        else:
            mask_array = mask

        if not np.any(mask_array):
            continue

        # Calculate object properties
        area = np.sum(mask_array)
        area_ratio = area / image_area

        # Robust depth score using eroded mask
        depth_score = get_robust_depth_score(mask_array, depth_map)

        # Calculate centroid and vertical position
        y_coords, x_coords = np.where(mask_array)
        if len(y_coords) == 0:
            continue

        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        bottom_y = np.max(y_coords)
        vertical_position = bottom_y / depth_map.shape[0]

        # Calculate mean color if image provided
        mean_color = None
        if img_rgb is not None:
            mask_pixels = img_rgb[mask_array]
            if len(mask_pixels) > 0:
                mean_color = np.mean(mask_pixels, axis=0)

        obj_info = {
            'id': i,
            'mask': mask_array,
            'depth_score': depth_score,
            'area': area,
            'area_ratio': area_ratio,
            'centroid': (centroid_x, centroid_y),
            'vertical_position': vertical_position,
            'bottom_y': bottom_y,
            'mean_color': mean_color,
            'label': labels[i] if labels and i < len(labels) else None
        }

        # Count touching masks
        dilated = ndimage.binary_dilation(mask_array, iterations=2)
        obj_info['num_contacts'] = 0
        for j, other_mask in enumerate(masks):
            if i != j:
                if isinstance(other_mask, dict):
                    other_array = other_mask['segmentation']
                else:
                    other_array = other_mask
                if np.any(dilated & other_array):
                    obj_info['num_contacts'] += 1

        all_objects.append(obj_info)

    if len(all_objects) == 0:
        return []

    # Step 1: Identify wall/background (largest mask)
    wall_idx = np.argmax([obj['area'] for obj in all_objects])
    wall = all_objects[wall_idx]

    # Step 2: Identify table/support (large, lower in image, many contacts)
    table_candidates = []
    for obj in all_objects:
        if obj['id'] != wall['id']:
            # Score based on: area, vertical position (lower = higher score), number of contacts
            table_score = (
                obj['area_ratio'] * 0.3 +  # Size matters
                obj['vertical_position'] * 0.4 +  # Lower in image
                min(obj['num_contacts'] / 5.0, 1.0) * 0.3  # Many contacts
            )
            table_candidates.append((obj, table_score))

    table = None
    if table_candidates:
        table_candidates.sort(key=lambda x: x[1], reverse=True)
        # Pick table if it has good score and is reasonably large
        if table_candidates[0][1] > 0.2 and table_candidates[0][0]['area_ratio'] > 0.02:
            table = table_candidates[0][0]

    # Step 3: Sort remaining objects by robust depth score
    remaining_objects = [obj for obj in all_objects
                         if obj['id'] != wall['id'] and
                         (table is None or obj['id'] != table['id'])]

    # Sort by depth score (lower = farther)
    remaining_objects.sort(key=lambda x: x['depth_score'])

    # Apply occlusion-based reordering
    for i in range(len(remaining_objects)):
        for j in range(i + 1, len(remaining_objects)):
            obj_i = remaining_objects[i]
            obj_j = remaining_objects[j]

            # Check if masks touch
            dilated_i = ndimage.binary_dilation(obj_i['mask'], iterations=2)
            dilated_j = ndimage.binary_dilation(obj_j['mask'], iterations=2)

            if np.any(dilated_i & dilated_j):
                # If j occludes i, swap them (j should come later)
                if check_occlusion(obj_j['mask'], obj_i['mask'], depth_map):
                    remaining_objects[i], remaining_objects[j] = remaining_objects[j], remaining_objects[i]

    # Break ties by area (bigger first)
    # Group objects with very similar depths
    depth_groups = []
    if remaining_objects:
        current_group = [remaining_objects[0]]
        for obj in remaining_objects[1:]:
            if abs(obj['depth_score'] - current_group[-1]['depth_score']) < 0.02:
                current_group.append(obj)
            else:
                # Sort current group by area before adding
                current_group.sort(key=lambda x: x['area'], reverse=True)
                depth_groups.append(current_group)
                current_group = [obj]

        # Add last group
        current_group.sort(key=lambda x: x['area'], reverse=True)
        depth_groups.append(current_group)

        # Flatten back to list
        remaining_objects = [obj for group in depth_groups for obj in group]

    # Step 4: Group into painting steps
    painting_steps = []

    # Step 1: Wall/background
    painting_steps.append({
        'step': 1,
        'group_name': 'wall/background',
        'mask_indices': [wall['id']],
        'objects': [wall]
    })

    # Step 2: Table/support (if exists)
    if table:
        painting_steps.append({
            'step': 2,
            'group_name': 'table/support',
            'mask_indices': [table['id']],
            'objects': [table]
        })

    # Steps 3+: Group remaining objects into 3 depth bands
    if remaining_objects:
        # Divide into 3 bands by depth
        n_objects = len(remaining_objects)
        if n_objects <= 3:
            # Each object gets its own step
            for i, obj in enumerate(remaining_objects):
                painting_steps.append({
                    'step': len(painting_steps) + 1,
                    'group_name': f'object_{obj["id"]}',
                    'mask_indices': [obj['id']],
                    'objects': [obj]
                })
        else:
            # Create 3 bands
            band_size = n_objects // 3
            bands = [
                remaining_objects[:band_size],  # Far
                remaining_objects[band_size:2*band_size],  # Mid
                remaining_objects[2*band_size:]  # Near
            ]
            band_names = ['far', 'mid', 'near']

            for band_idx, (band, band_name) in enumerate(zip(bands, band_names)):
                if not band:
                    continue

                # Within each band, merge similar objects
                merged_groups = []
                used = set()

                for obj in band:
                    if obj['id'] in used:
                        continue

                    # Start new group with this object
                    group = [obj]
                    used.add(obj['id'])

                    # Find similar objects in same band
                    for other_obj in band:
                        if other_obj['id'] in used:
                            continue

                        # Check similarity
                        similar = False

                        # Same label?
                        if obj['label'] and other_obj['label'] and obj['label'] == other_obj['label']:
                            similar = True
                        # Similar size and color?
                        elif obj['mean_color'] is not None and other_obj['mean_color'] is not None:
                            size_ratio = min(obj['area'], other_obj['area']) / max(obj['area'], other_obj['area'])
                            color_dist = np.linalg.norm(obj['mean_color'] - other_obj['mean_color'])
                            if size_ratio > 0.3 and color_dist < 50:
                                similar = True

                        if similar:
                            group.append(other_obj)
                            used.add(other_obj['id'])

                    merged_groups.append(group)

                # Add each merged group as a step
                for group in merged_groups:
                    indices = [obj['id'] for obj in group]
                    labels_str = ', '.join([str(obj['label']) for obj in group if obj['label']]) or band_name

                    painting_steps.append({
                        'step': len(painting_steps) + 1,
                        'group_name': f'{band_name}_band: {labels_str}' if labels_str else band_name,
                        'mask_indices': indices,
                        'objects': group
                    })

    return painting_steps


def visualize_painting_steps(img_bgr, depth_map, painting_steps, masks):
    """
    Visualize the painting steps with grouped objects.
    Shows original, depth map, and step-by-step visualization.
    """
    if isinstance(img_bgr, np.ndarray) and len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr

    # Calculate number of columns needed
    n_steps = len(painting_steps)
    n_cols = min(4, max(3, n_steps + 1))  # At least 3 cols, max 4
    n_rows = (n_steps + 2) // n_cols + 1  # +2 for original and legend

    # Create figure
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    # a) Original image
    ax = plt.subplot(n_rows, n_cols, 1)
    ax.imshow(img_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # b) Panel per step: overlay masks with semi-transparent colors
    colors = plt.cm.Set3(np.linspace(0, 1, max(12, n_steps)))

    for step_idx, step in enumerate(painting_steps):
        ax = plt.subplot(n_rows, n_cols, step_idx + 2)

        # Start with original image
        step_img = img_rgb.copy().astype(np.float32)
        overlay = np.zeros_like(step_img)

        # Get color for this step
        step_color = colors[step_idx % len(colors)][:3]

        # Create combined mask for all objects in this step
        combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        for obj in step['objects']:
            combined_mask |= obj['mask']

        # Apply semi-transparent color overlay
        for c in range(3):
            overlay[:, :, c][combined_mask] = step_color[c] * 255

        # Blend with original (0.6 opacity)
        step_img[combined_mask] = step_img[combined_mask] * 0.4 + overlay[combined_mask] * 0.6

        # Draw thin outlines
        from skimage import segmentation
        for obj in step['objects']:
            boundaries = segmentation.find_boundaries(obj['mask'], mode='outer')
            step_img[boundaries] = [255, 255, 255]  # White outline

        ax.imshow(step_img.astype(np.uint8))

        # Title with step info
        indices_str = ', '.join([str(obj['id']) for obj in step['objects']])
        title = f"Step {step['step']}: {step['group_name']}\n"
        title += f"Masks: [{indices_str}]"
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # c) Small legend
    ax_legend = plt.subplot(n_rows, n_cols, n_rows * n_cols)
    ax_legend.axis('off')

    legend_text = "PAINTING STEPS\n" + "="*20 + "\n\n"
    for step in painting_steps:
        indices_str = ', '.join([str(idx) for idx in step['mask_indices']])
        legend_text += f"Step {step['step']}: {step['group_name']}\n"
        legend_text += f"  Masks: [{indices_str}]\n"
        legend_text += f"  Objects: {len(step['objects'])}\n\n"

    ax_legend.text(0.1, 0.9, legend_text, fontsize=10,
                   verticalalignment='top', transform=ax_legend.transAxes,
                   family='monospace')

    plt.suptitle(f'Robust Depth-Based Painting Steps - {len(painting_steps)} Steps',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Print steps list
    print("\nPainting Steps (ordered back to front):\n" + "="*50)
    for step in painting_steps:
        print(f"Step {step['step']}: {step['group_name']}")
        print(f"  Mask indices: {step['mask_indices']}")
        print(f"  Number of objects: {len(step['objects'])}")
        if step['objects']:
            areas = [obj['area_ratio'] * 100 for obj in step['objects']]
            print(f"  Areas: {', '.join([f'{a:.1f}%' for a in areas])}")
        print()

    return fig


# Create widget interface
output = widgets.Output()

# Controls
use_labels_checkbox = widgets.Checkbox(
    value=False,
    description='Use Object Labels',
    style={'description_width': 'initial'},
    tooltip='Use object labels for grouping if available'
)

export_steps_checkbox = widgets.Checkbox(
    value=False,
    description='Export Steps Data',
    style={'description_width': 'initial'}
)

run_button = widgets.Button(
    description='Create Robust Painting Steps',
    button_style='success',
    tooltip='Create robust depth-ordered painting steps'
)

status_text = widgets.HTML(value='<b>Ready to create painting steps</b>')

def on_create_steps(b):
    """Handle step creation button click"""
    global painting_steps_result

    with output:
        clear_output(wait=True)

        # Check prerequisites
        if 'pipeline_results' not in globals():
            status_text.value = '<b style="color: red;">Pipeline results not found!</b>'
            return

        if 'masks' not in pipeline_results or 'depth_map' not in pipeline_results:
            status_text.value = '<b style="color: red;">Missing masks or depth map!</b>'
            return

        status_text.value = '<b style="color: blue;">Creating robust painting steps...</b>'

        try:
            # Prepare labels if requested
            labels = None
            if use_labels_checkbox.value and 'labels' in pipeline_results:
                labels = pipeline_results['labels']

            # Create painting steps with robust ordering
            painting_steps_result = create_painting_steps(
                pipeline_results['masks'],
                pipeline_results['depth_map'],
                img_rgb=sample_img if 'sample_img' in globals() else None,
                labels=labels
            )

            print(f"Created {len(painting_steps_result)} Robust Painting Steps")
            print("="*60)

            # Visualize
            fig = visualize_painting_steps(
                sample_img,
                pipeline_results['depth_map'],
                painting_steps_result,
                pipeline_results['masks']
            )
            plt.show()

            # Summary already printed by visualize_painting_steps

            # Export if requested
            if export_steps_checkbox.value:
                import json

                # Export steps as list of dicts
                export_data = []
                for step in painting_steps_result:
                    export_data.append({
                        'step': step['step'],
                        'group_name': step['group_name'],
                        'mask_indices': step['mask_indices']
                    })

                with open('painting_steps.json', 'w') as f:
                    json.dump(export_data, f, indent=2)

                fig.savefig('painting_steps_visualization.png', dpi=150, bbox_inches='tight')

                print("\nExported:")
                print("   • painting_steps.json - Steps data")
                print("   • painting_steps_visualization.png - Visual guide")

            # Store in pipeline
            if 'pipeline_results' in globals():
                pipeline_results['painting_steps'] = painting_steps_result

            status_text.value = f'<b style="color: green;">Created {len(painting_steps_result)} robust painting steps!</b>'

            print("\nPainting steps ready!")
            print("   1. Wall/background first")
            print("   2. Table/support second")
            print("   3+ Remaining objects grouped by depth")

        except Exception as e:
            status_text.value = f'<b style="color: red;">❌ Error: {str(e)}</b>'
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

# Connect handler
run_button.on_click(on_create_steps)

# Layout
controls = widgets.VBox([
    widgets.HTML('<h3>Robust Depth-Based Painting Steps</h3>'),
    use_labels_checkbox,
    export_steps_checkbox,
    run_button,
    status_text
], layout=widgets.Layout(padding='10px', border='2px solid #ccc', border_radius='10px'))

display(controls)
display(output)

print("Robust Painting Steps Widget Ready!")
print("="*60)
print("This widget creates robust depth-ordered painting steps:")
print("  • Wall/background first (largest mask)")
print("  • Table/support second (lower, many contacts)")
print("  • Remaining objects sorted by robust depth (median of eroded mask)")
print("  • Occlusion-aware ordering for touching objects")
print("  • Groups similar objects within depth bands")
print("\nPrerequisites:")
print("  1. Run SAM 2.1 detection")
print("  2. Run DepthAnything V2")