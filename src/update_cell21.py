#!/usr/bin/env python3
"""
Script to enhance Cell 21 of MasterStroke.ipynb with mask completion functionality
"""

import json
import sys

# The complete_occluded_masks function
COMPLETE_OCCLUDED_MASKS_FUNC = '''
def complete_occluded_masks(ordered_objects, depth_map, img_shape,
                            expansion_strength=1.0, depth_tolerance=0.15):
    """
    Complete occluded portions of masks using depth-guided expansion.
    Background objects are expanded to include regions hidden by foreground objects.

    Args:
        ordered_objects: List from calculate_object_order (background→foreground)
        depth_map: Normalized depth (DepthAnything: higher values = closer/foreground)
        img_shape: (height, width)
        expansion_strength: Multiplier for dilation iterations (0-3)
        depth_tolerance: Max depth difference for expansion (0.05-0.5)

    Returns:
        List of objects with original_mask and completed_mask fields
    """
    from scipy import ndimage

    h, w = img_shape
    completed_objects = []

    # Track which pixels are "claimed" by each depth layer
    # This prevents background objects from expanding into foreground space
    claimed_mask = np.zeros((h, w), dtype=bool)

    # Process in drawing order (background to foreground)
    for obj_idx, obj in enumerate(ordered_objects):
        original_mask = obj['mask_data']['mask'].copy()
        obj_depth = obj['avg_depth']
        original_area = original_mask.sum()

        # Find occluding objects (drawn after this one = closer)
        # DepthAnything: higher depth = closer, so occluders have depth > obj_depth
        occluding_masks = []
        occluded_by_indices = []

        for other_idx, other_obj in enumerate(ordered_objects):
            if other_obj['draw_order'] > obj['draw_order']:
                # Check if they overlap (potential occlusion)
                overlap = np.logical_and(original_mask, other_obj['mask_data']['mask'])
                if overlap.sum() > 0:
                    # Verify depth relationship (other is closer = higher depth)
                    if other_obj['avg_depth'] > obj_depth:
                        occluding_masks.append(other_obj['mask_data']['mask'])
                        occluded_by_indices.append(other_obj['index'])

        # Create expansion constraint mask
        # Can expand into: regions with similar depth, not occupied by occluders
        occluder_union = np.zeros_like(original_mask)
        for occ_mask in occluding_masks:
            occluder_union |= occ_mask

        # Depth similarity mask (where depth is close to object's depth)
        # Note: DepthAnything has higher=closer, so we want similar absolute values
        depth_similar = np.abs(depth_map - obj_depth) < depth_tolerance

        # Valid expansion zone: similar depth AND not occluded AND not claimed
        valid_expansion_zone = depth_similar & (~occluder_union) & (~claimed_mask)

        # Iterative expansion
        completed_mask = original_mask.copy()
        max_iterations = int(20 * expansion_strength)  # Scale by strength

        for iteration in range(max_iterations):
            # Dilate by 1 pixel
            dilated = ndimage.binary_dilation(completed_mask, iterations=1)

            # Only keep expansion in valid zone
            expanded_region = dilated & valid_expansion_zone & (~completed_mask)

            if expanded_region.sum() == 0:
                break  # No more expansion possible

            # Add expanded region to mask
            completed_mask |= expanded_region

            # Safety check: cap maximum expansion at 3.0x original area
            if completed_mask.sum() > original_area * 3.0:
                break

        # Mark this object's completed mask as claimed
        claimed_mask |= completed_mask

        # Calculate statistics
        completed_area = completed_mask.sum()
        expansion_ratio = completed_area / original_area if original_area > 0 else 1.0

        # Store results
        obj_copy = obj.copy()
        obj_copy['original_mask'] = original_mask
        obj_copy['completed_mask'] = completed_mask
        obj_copy['completed_area'] = completed_area
        obj_copy['expansion_ratio'] = expansion_ratio
        obj_copy['occluded_by_indices'] = occluded_by_indices

        # Update mask_data to use completed mask for downstream operations
        obj_copy['mask_data'] = obj['mask_data'].copy()
        obj_copy['mask_data']['mask'] = completed_mask
        obj_copy['area'] = completed_area

        completed_objects.append(obj_copy)

    return completed_objects
'''

# The visualize_mask_completion function
VISUALIZE_MASK_COMPLETION_FUNC = '''
def visualize_mask_completion(ordered_objects, depth_map, img_rgb):
    """Visualize original vs completed masks with expansion analysis"""
    num_objects = len(ordered_objects)

    if num_objects == 0:
        print("No objects to visualize")
        return

    # Filter to objects with significant expansion
    expanded_objects = [obj for obj in ordered_objects if obj.get('expansion_ratio', 1.0) > 1.05]

    if len(expanded_objects) == 0:
        print("No objects required expansion (all are complete)")
        return

    # Create 4-panel figure
    fig = plt.figure(figsize=(20, 12))

    # 1. Original Masks
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(img_rgb, alpha=0.3)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_objects))
    for obj in ordered_objects:
        mask = obj['original_mask']
        overlay = np.zeros((*mask.shape, 4))
        color = colors[obj['draw_order'] - 1]
        overlay[mask] = color
        ax1.imshow(overlay, alpha=0.7)
    ax1.set_title('Original SAM Masks (Visible Portions Only)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Completed Masks
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(img_rgb, alpha=0.3)
    for obj in ordered_objects:
        mask = obj['completed_mask']
        overlay = np.zeros((*mask.shape, 4))
        color = colors[obj['draw_order'] - 1]
        overlay[mask] = color
        ax2.imshow(overlay, alpha=0.7)
    ax2.set_title('Completed Masks (Full Objects)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Expansion Heatmap
    ax3 = plt.subplot(2, 2, 3)
    heatmap = np.zeros(depth_map.shape)
    for obj in ordered_objects:
        expansion_only = obj['completed_mask'] & (~obj['original_mask'])
        heatmap[expansion_only] = obj['expansion_ratio']

    im = ax3.imshow(heatmap, cmap='RdYlGn_r', vmin=1.0, vmax=2.5)
    ax3.set_title('Expansion Heatmap (Color = Expansion Ratio)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, label='Expansion Ratio')

    # 4. Statistics Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    table_data = []
    for obj in expanded_objects[:15]:  # Show first 15 expanded objects
        occluders = f"#{', #'.join(map(str, [o+1 for o in obj['occluded_by_indices']]))}" if obj['occluded_by_indices'] else "None"
        table_data.append([
            f"#{obj['draw_order']}",
            f"{obj['area']:.0f}",
            f"{obj['completed_area']:.0f}",
            f"+{(obj['expansion_ratio']-1)*100:.1f}%",
            occluders
        ])

    table = ax4.table(cellText=table_data,
                     colLabels=['Order', 'Original\nArea', 'Completed\nArea', 'Expansion', 'Occluded By'],
                     cellLoc='center',
                     loc='center',
                     colColours=['lightblue']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)
    ax4.set_title(f'Mask Completion Statistics ({len(expanded_objects)} objects expanded)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f'Mask Completion Analysis - {num_objects} Total Objects',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # Print summary
    expansion_ratios = [obj['expansion_ratio'] for obj in ordered_objects]
    print(f"\\n{'='*80}")
    print(f"MASK COMPLETION SUMMARY")
    print(f"{'='*80}")
    print(f"Total objects: {num_objects}")
    print(f"Objects expanded: {len(expanded_objects)} ({len(expanded_objects)/num_objects*100:.1f}%)")
    print(f"Average expansion ratio: {np.mean(expansion_ratios):.2f}x")
    print(f"Max expansion ratio: {np.max(expansion_ratios):.2f}x")
    print(f"{'='*80}\\n")
'''

# New widgets to add
NEW_WIDGETS = '''
complete_masks_checkbox = widgets.Checkbox(
    value=True,
    description='Complete Occluded Masks',
    style={'description_width': 'initial'},
    tooltip='Expand masks to include occluded portions'
)

expansion_strength = widgets.FloatSlider(
    value=1.0,
    min=0.0,
    max=3.0,
    step=0.1,
    description='Expansion Strength:',
    style={'description_width': 'initial'},
    tooltip='Expansion aggressiveness (0=none, 3=very aggressive)'
)

depth_tolerance = widgets.FloatSlider(
    value=0.15,
    min=0.05,
    max=0.5,
    step=0.05,
    description='Depth Tolerance:',
    style={'description_width': 'initial'},
    tooltip='Depth similarity for expansion (lower=stricter)'
)

show_completion_analysis = widgets.Checkbox(
    value=False,
    description='Show Completion Analysis',
    style={'description_width': 'initial'},
    tooltip='Visualize original vs completed masks'
)
'''

def main():
    # Read notebook
    with open('/teamspace/studios/this_studio/sam2/MasterStroke.ipynb', 'r') as f:
        nb = json.load(f)

    # Get Cell 21 source
    source = '\\n'.join(nb['cells'][21]['source'])
    lines = source.split('\\n')

    print(f"Original cell has {len(lines)} lines")

    # Step 1: Insert complete_occluded_masks() after calculate_object_order (before visualize_step_by_step)
    for i, line in enumerate(lines):
        if 'def visualize_step_by_step' in line:
            insert_pos_1 = i
            break

    # Step 2: Insert visualize_mask_completion() before on_run_clicked
    for i, line in enumerate(lines):
        if 'def on_run_clicked' in line:
            insert_pos_2 = i
            break

    # Step 3: Find where to insert new widgets (after export_order widget, before status_text)
    widget_insert_pos = None
    for i, line in enumerate(lines):
        if 'status_text = widgets.HTML' in line:
            widget_insert_pos = i
            break

    # Step 4: Find widget layout position
    layout_pos = None
    for i, line in enumerate(lines):
        if 'controls = widgets.VBox' in line:
            layout_pos = i
            break

    print(f"Insertion positions:")
    print(f"  complete_occluded_masks: line {insert_pos_1}")
    print(f"  visualize_mask_completion: line {insert_pos_2}")
    print(f"  new widgets: line {widget_insert_pos}")
    print(f"  layout update: line {layout_pos}")

    # Build new source
    new_lines = []

    # Add lines up to visualize_step_by_step
    new_lines.extend(lines[:insert_pos_1])

    # Add complete_occluded_masks function
    new_lines.append(COMPLETE_OCCLUDED_MASKS_FUNC)

    # Add lines from visualize_step_by_step to on_run_clicked
    new_lines.extend(lines[insert_pos_1:insert_pos_2])

    # Add visualize_mask_completion function
    new_lines.append(VISUALIZE_MASK_COMPLETION_FUNC)

    # Add lines from on_run_clicked to widget insertion point
    new_lines.extend(lines[insert_pos_2:widget_insert_pos])

    # Add new widgets
    new_lines.append(NEW_WIDGETS)

    # Add status_text and output_area
    new_lines.extend(lines[widget_insert_pos:layout_pos])

    # Now we need to update the layout - find and modify it
    # We'll manually construct the new layout
    layout_section_found = False
    i = layout_pos
    while i < len(lines):
        if not layout_section_found and 'controls = widgets.VBox' in lines[i]:
            layout_section_found = True
            # Skip to the closing ])
            new_lines.append("controls = widgets.VBox([")
            new_lines.append("    widgets.HTML('<h3>🎨 Object Drawing Order Analysis - Background to Foreground</h3>'),")
            new_lines.append("    widgets.HTML('<p>Orders objects from background (far) to foreground (near) for proper painting sequence.</p>'),")
            new_lines.append("    widgets.HBox([depth_weight, size_weight]),")
            new_lines.append("    widgets.HBox([num_steps, show_step_frames]),")
            new_lines.append("    widgets.HBox([show_visualization, export_order]),")
            new_lines.append("    widgets.HTML('<hr><h4>Mask Completion Settings</h4>'),")
            new_lines.append("    complete_masks_checkbox,")
            new_lines.append("    widgets.HBox([expansion_strength, depth_tolerance]),")
            new_lines.append("    show_completion_analysis,")
            new_lines.append("    run_button,")
            new_lines.append("    status_text")
            new_lines.append("])")

            # Skip original layout definition
            while i < len(lines) and not lines[i].strip().startswith('display('):
                i += 1
            i -= 1  # Back up one so we include display in next iteration
        elif layout_section_found and 'display(controls' in lines[i]:
            new_lines.extend(lines[i:])
            break
        i += 1

    # Write back to notebook
    new_source = '\\n'.join(new_lines)
    nb['cells'][21]['source'] = new_source.split('\\n')

    with open('/teamspace/studios/this_studio/sam2/MasterStroke.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\\nUpdated cell now has {len(new_lines)} lines")
    print("✅ Cell 21 updated with mask completion functions and widgets!")

if __name__ == '__main__':
    main()
