# @title Step 6: Depth-Based Object Ordering
"""
Order SAM-detected objects based on their median depth values from DepthAnything V2
Visualize the depth-based drawing order for artistic rendering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display, clear_output

def analyze_objects_by_depth(masks, depth_map, depth_threshold=0.05, min_step_size=0.02):
    """
    Analyze each SAM mask with depth information and create meaningful depth steps
    Merges smaller objects into larger ones at similar depths to create fewer steps

    Args:
        masks: List of SAM masks
        depth_map: Normalized depth map from DepthAnything V2
        depth_threshold: Threshold for grouping objects of similar depth (default 0.05)
        min_step_size: Minimum size for an object to create its own step (default 0.02 = 2% of image)

    Returns:
        List of depth steps, each containing multiple objects at similar depths
    """
    depth_objects = []

    # First, extract all valid objects
    for i, mask in enumerate(masks):
        if isinstance(mask, dict):
            mask_array = mask['segmentation']
        else:
            mask_array = mask

        # Skip empty masks
        if not np.any(mask_array):
            continue

        # Get depth values within the mask
        masked_depths = depth_map[mask_array]

        if len(masked_depths) == 0:
            continue

        # Calculate depth statistics
        obj_info = {
            'id': i,
            'mask': mask_array,
            'median_depth': np.median(masked_depths),
            'mean_depth': np.mean(masked_depths),
            'min_depth': np.min(masked_depths),
            'max_depth': np.max(masked_depths),
            'std_depth': np.std(masked_depths),
            'area': np.sum(mask_array),
            'area_ratio': np.sum(mask_array) / (depth_map.shape[0] * depth_map.shape[1])
        }

        # Calculate centroid
        y_coords, x_coords = np.where(mask_array)
        if len(y_coords) > 0:
            obj_info['centroid'] = (np.mean(x_coords), np.mean(y_coords))
        else:
            obj_info['centroid'] = (0, 0)

        # Classify depth layer (higher values = closer/foreground)
        if obj_info['median_depth'] > 0.7:
            obj_info['layer'] = 'Foreground'
            obj_info['layer_color'] = '#FF6B6B'  # Red
        elif obj_info['median_depth'] > 0.4:
            obj_info['layer'] = 'Midground'
            obj_info['layer_color'] = '#4ECDC4'  # Teal
        else:
            obj_info['layer'] = 'Background'
            obj_info['layer_color'] = '#45B7D1'  # Blue

        depth_objects.append(obj_info)

    # Sort by depth first
    depth_objects.sort(key=lambda x: x['median_depth'])

    # Create depth steps using clustering approach
    if len(depth_objects) > 0:
        # Identify anchor objects (large objects that define steps)
        anchor_objects = [obj for obj in depth_objects if obj['area_ratio'] >= min_step_size]

        if not anchor_objects:
            # If no large objects, use the largest available
            anchor_objects = sorted(depth_objects, key=lambda x: x['area'], reverse=True)[:min(5, len(depth_objects))]

        # Create steps around anchor objects
        depth_steps = []
        assigned = [False] * len(depth_objects)

        for anchor in anchor_objects:
            if assigned[depth_objects.index(anchor)]:
                continue

            # Create a new step centered on this anchor
            step_objects = []
            anchor_depth = anchor['median_depth']

            # Add all objects within threshold of this anchor
            for i, obj in enumerate(depth_objects):
                if not assigned[i] and abs(obj['median_depth'] - anchor_depth) <= depth_threshold:
                    step_objects.append(obj)
                    assigned[i] = True

            if step_objects:
                # Sort objects within step by area (larger first)
                step_objects.sort(key=lambda x: x['area'], reverse=True)
                depth_steps.append(step_objects)

        # Handle any unassigned objects (merge with nearest step)
        for i, obj in enumerate(depth_objects):
            if not assigned[i]:
                # Find nearest step
                best_step = 0
                best_distance = float('inf')

                for step_idx, step in enumerate(depth_steps):
                    avg_depth = np.mean([o['median_depth'] for o in step])
                    distance = abs(obj['median_depth'] - avg_depth)
                    if distance < best_distance:
                        best_distance = distance
                        best_step = step_idx

                if depth_steps:
                    depth_steps[best_step].append(obj)
                    depth_steps[best_step].sort(key=lambda x: x['area'], reverse=True)

        # Sort steps by average depth (background to foreground)
        depth_steps.sort(key=lambda step: np.mean([obj['median_depth'] for obj in step]))

        # Flatten and assign step numbers and drawing order
        sorted_objects = []
        draw_order = 1

        for step_num, step_objects in enumerate(depth_steps, 1):
            for obj in step_objects:
                obj['step'] = step_num
                obj['draw_order'] = draw_order

                # Color gradient from blue (far/background) to red (near/foreground)
                step_ratio = (step_num - 1) / max(len(depth_steps) - 1, 1)
                obj['step_color'] = plt.cm.RdYlBu_r(step_ratio)

                sorted_objects.append(obj)
                draw_order += 1

        return sorted_objects

    return depth_objects

def visualize_depth_ordering(img_bgr, depth_map, depth_objects, num_display=20):
    """
    Create comprehensive visualization of depth-based object ordering

    Args:
        img_bgr: Original image in BGR format
        depth_map: Depth map from DepthAnything V2
        depth_objects: List of objects with depth information
        num_display: Number of objects to display in detail
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Create figure with multiple subplots - larger for more previews
    fig = plt.figure(figsize=(28, 20))

    # Use GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 6, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Depth map
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(depth_map, cmap='turbo')
    ax2.set_title('Depth Map\n(Blue=Far, Red=Near)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # 3. Segmentation overlay with depth ordering numbers and step groupings
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = np.zeros_like(img_rgb)

    # Create color map for objects based on step grouping
    for obj in depth_objects[:num_display]:
        # Use step color for consistency
        if 'step_color' in obj:
            color = obj['step_color'][:3]
        else:
            hue = (obj['draw_order'] - 1) / max(len(depth_objects), 1)
            color = plt.cm.hsv(hue)[:3]
        overlay[obj['mask']] = (np.array(color) * 255).astype(np.uint8)

    # Blend with original
    blended = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)
    ax3.imshow(blended)

    # Add order numbers with step info
    for obj in depth_objects[:num_display]:
        cx, cy = obj['centroid']
        # Show both order and step
        label = f"{obj['draw_order']}\n(S{obj['step']})"
        ax3.text(cx, cy, label,
                fontsize=8, color='white', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax3.set_title(f'Object Ordering by Steps\n({num_display} objects, Steps shown as S#)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Depth layers visualization
    ax4 = fig.add_subplot(gs[0, 3])
    layer_viz = np.zeros_like(img_rgb)

    layer_colors = {
        'Foreground': [255, 107, 107],  # Red
        'Midground': [78, 205, 196],    # Teal
        'Background': [69, 183, 209]     # Blue
    }

    for obj in depth_objects:
        layer_viz[obj['mask']] = layer_colors[obj['layer']]

    ax4.imshow(layer_viz)
    ax4.set_title('Depth Layers\n(Red=Front, Teal=Mid, Blue=Back)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. Drawing order sequence - expanded to show more objects
    ax5 = fig.add_subplot(gs[0:2, 4:6])  # Span 2 rows, 2 columns
    ax5.axis('off')

    # Create a bar chart showing object depths
    top_n = min(30, len(depth_objects))  # Show up to 30 objects
    orders = [obj['draw_order'] for obj in depth_objects[:top_n]]
    depths = [obj['median_depth'] for obj in depth_objects[:top_n]]
    steps = [obj['step'] for obj in depth_objects[:top_n]]
    colors = [obj['layer_color'] for obj in depth_objects[:top_n]]

    bars = ax5.barh(range(top_n), depths, color=colors, alpha=0.7)
    ax5.set_ylim(-0.5, top_n - 0.5)
    ax5.set_xlim(0, 1)
    ax5.set_xlabel('Median Depth', fontsize=10)
    ax5.set_title(f'Top {top_n} Objects by Order', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()

    # Add order and step labels
    for i, (order, depth, step) in enumerate(zip(orders, depths, steps)):
        ax5.text(-0.08, i, f'#{order} S{step}', ha='right', va='center', fontsize=7, fontweight='bold')
        ax5.text(depth + 0.02, i, f'{depth:.3f}', ha='left', va='center', fontsize=6)

    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # 6. Size vs Depth scatter plot
    ax6 = fig.add_subplot(gs[1, 0])
    sizes = [obj['area_ratio'] * 100 for obj in depth_objects]  # Convert to percentage
    depths_all = [obj['median_depth'] for obj in depth_objects]
    colors_all = [obj['layer_color'] for obj in depth_objects]

    scatter = ax6.scatter(depths_all, sizes, c=colors_all, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Median Depth', fontsize=10)
    ax6.set_ylabel('Object Size (%)', fontsize=10)
    ax6.set_title('Object Size vs Depth', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 1)

    # Add layer boundaries
    ax6.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
    ax6.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)

    # Individual object previews - show up to 18 objects (3 rows x 6 columns)
    preview_count = min(18, len(depth_objects), num_display)

    for idx in range(preview_count):
        # Calculate position in grid - starting from row 2
        row = idx // 6 + 2
        col = idx % 6
        ax = fig.add_subplot(gs[row, col])

        obj = depth_objects[idx]

        # Create masked image
        masked_img = img_rgb.copy()
        mask_inv = ~obj['mask']
        masked_img[mask_inv] = masked_img[mask_inv] * 0.2  # Darken non-masked areas

        ax.imshow(masked_img)
        ax.set_title(f"#{obj['draw_order']} S{obj['step']}\nD:{obj['median_depth']:.2f}",
                    fontsize=9)
        ax.axis('off')

        # Add bounding box
        y_coords, x_coords = np.where(obj['mask'])
        if len(y_coords) > 0:
            bbox = Rectangle((min(x_coords), min(y_coords)),
                           max(x_coords) - min(x_coords),
                           max(y_coords) - min(y_coords),
                           linewidth=1.5, edgecolor=obj['layer_color'],
                           facecolor='none', alpha=0.8)
            ax.add_patch(bbox)

    # Depth histogram - move to different location
    ax11 = fig.add_subplot(gs[1, 1])
    all_depths = [obj['median_depth'] for obj in depth_objects]

    n, bins, patches = ax11.hist(all_depths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')

    # Color bars by layer
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.3:
            patch.set_facecolor('#FF6B6B')
        elif bin_center < 0.6:
            patch.set_facecolor('#4ECDC4')
        else:
            patch.set_facecolor('#45B7D1')

    ax11.set_xlabel('Depth Value', fontsize=10)
    ax11.set_ylabel('Number of Objects', fontsize=10)
    ax11.set_title('Object Depth Distribution', fontsize=11, fontweight='bold')
    ax11.grid(True, alpha=0.3)

    # Layer statistics pie chart
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis('off')

    # Count objects per layer
    layer_counts = {'Foreground': 0, 'Midground': 0, 'Background': 0}
    for obj in depth_objects:
        layer_counts[obj['layer']] += 1

    # Create pie chart
    if sum(layer_counts.values()) > 0:
        wedges, texts, autotexts = ax12.pie(layer_counts.values(),
                                             labels=layer_counts.keys(),
                                             colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                             autopct='%1.0f%%',
                                             startangle=90)

        # Enhance text
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

    ax12.set_title('Layer Distribution', fontsize=11, fontweight='bold')

    # Add step summary in remaining space
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.axis('off')

    # Create step summary text
    step_counts = {}
    for obj in depth_objects:
        if obj['step'] not in step_counts:
            step_counts[obj['step']] = 0
        step_counts[obj['step']] += 1

    summary_text = "Step Summary:\n" + "-"*20 + "\n"
    for step in sorted(step_counts.keys())[:5]:  # Show first 5 steps
        step_objs = [o for o in depth_objects if o['step'] == step]
        avg_depth = np.mean([o['median_depth'] for o in step_objs])
        summary_text += f"Step {step}: {step_counts[step]} objects\n"
        summary_text += f"  Avg depth: {avg_depth:.3f}\n"

    ax13.text(0.1, 0.9, summary_text, fontsize=9, verticalalignment='top',
              transform=ax13.transAxes, family='monospace')
    ax13.set_title('Step Groups', fontsize=11, fontweight='bold')

    # Main title
    plt.suptitle(f'Depth-Based Object Ordering Analysis\n{len(depth_objects)} Objects → {len(step_counts)} Steps (Showing {preview_count} previews)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return fig

# Create interactive widget interface
output = widgets.Output()

# Number of objects to display
num_display_slider = widgets.IntSlider(
    value=20,
    min=5,
    max=50,
    step=5,
    description='Display Objects:',
    style={'description_width': 'initial'},
    tooltip='Number of objects to show in visualization'
)

# Depth threshold for grouping
depth_threshold_slider = widgets.FloatSlider(
    value=0.05,
    min=0.01,
    max=0.2,
    step=0.01,
    description='Depth Threshold:',
    style={'description_width': 'initial'},
    tooltip='Threshold for grouping objects of similar depth into steps'
)

# Minimum size for anchor objects
min_size_slider = widgets.FloatSlider(
    value=0.02,
    min=0.005,
    max=0.1,
    step=0.005,
    description='Min Anchor Size:',
    style={'description_width': 'initial'},
    tooltip='Minimum size (% of image) for an object to anchor a step'
)

# Export options
export_order_checkbox = widgets.Checkbox(
    value=False,
    description='Export Ordering Data',
    style={'description_width': 'initial'}
)

show_stats_checkbox = widgets.Checkbox(
    value=True,
    description='Show Detailed Stats',
    style={'description_width': 'initial'}
)

# Run button
run_button = widgets.Button(
    description='🎨 Analyze Depth Ordering',
    button_style='success',
    tooltip='Order objects based on depth'
)

# Status text
status_text = widgets.HTML(value='<b>Ready to analyze object ordering</b>')

def on_run_ordering(b):
    """Handle ordering analysis button click"""
    global depth_ordered_objects

    with output:
        clear_output(wait=True)

        # Check prerequisites
        if 'pipeline_results' not in globals():
            status_text.value = '<b style="color: red;">❌ Pipeline results not found! Run SAM detection first.</b>'
            return

        if 'masks' not in pipeline_results:
            status_text.value = '<b style="color: red;">❌ No masks found! Run SAM detection first.</b>'
            return

        if 'depth_map' not in pipeline_results:
            status_text.value = '<b style="color: red;">❌ No depth map found! Run DepthAnything V2 first.</b>'
            return

        status_text.value = '<b style="color: blue;">🔄 Analyzing depth-based ordering...</b>'

        try:
            # Analyze objects with clustering approach
            depth_ordered_objects = analyze_objects_by_depth(
                pipeline_results['masks'],
                pipeline_results['depth_map'],
                depth_threshold=depth_threshold_slider.value,
                min_step_size=min_size_slider.value
            )

            # Count unique steps
            unique_steps = len(set(obj['step'] for obj in depth_ordered_objects)) if depth_ordered_objects else 0

            print(f"📊 Analyzed {len(depth_ordered_objects)} objects → Grouped into {unique_steps} steps")
            print(f"   Depth threshold: {depth_threshold_slider.value:.3f}")
            print(f"   Min anchor size: {min_size_slider.value*100:.1f}% of image")
            print("="*60)

            # Create visualization
            fig = visualize_depth_ordering(
                sample_img,
                pipeline_results['depth_map'],
                depth_ordered_objects,
                num_display=num_display_slider.value
            )

            # Show detailed statistics
            if show_stats_checkbox.value:
                print("\n📈 Depth Ordering Statistics:")
                print("="*60)

                # Layer summary
                layer_counts = {'Foreground': 0, 'Midground': 0, 'Background': 0}
                total_area = {'Foreground': 0, 'Midground': 0, 'Background': 0}

                for obj in depth_ordered_objects:
                    layer_counts[obj['layer']] += 1
                    total_area[obj['layer']] += obj['area_ratio']

                print("\n🎨 Layer Distribution:")
                for layer in ['Background', 'Midground', 'Foreground']:
                    count = layer_counts[layer]
                    area = total_area[layer] * 100
                    print(f"  {layer:12s}: {count:3d} objects ({area:5.1f}% area)")

                # Top objects by size
                print("\n📏 Largest Objects (by area):")
                sorted_by_size = sorted(depth_ordered_objects, key=lambda x: x['area_ratio'], reverse=True)
                for i, obj in enumerate(sorted_by_size[:5]):
                    print(f"  {i+1}. Order #{obj['draw_order']:3d} - "
                          f"Size: {obj['area_ratio']*100:5.2f}% - "
                          f"Depth: {obj['median_depth']:.3f} - "
                          f"{obj['layer']}")

                # Step grouping information
                print("\n🎯 Depth Steps (Background → Foreground):")
                print("   Each step contains objects at similar depth to paint together")

                # Collect step information
                step_info = {}
                for obj in depth_ordered_objects:
                    if obj['step'] not in step_info:
                        step_info[obj['step']] = {
                            'objects': [],
                            'depths': [],
                            'sizes': [],
                            'total_area': 0
                        }
                    step_info[obj['step']]['objects'].append(obj)
                    step_info[obj['step']]['depths'].append(obj['median_depth'])
                    step_info[obj['step']]['sizes'].append(obj['area_ratio'])
                    step_info[obj['step']]['total_area'] += obj['area_ratio']

                # Print step summaries
                for step_num in sorted(step_info.keys()):
                    info = step_info[step_num]
                    depth_range = f"{min(info['depths']):.3f}-{max(info['depths']):.3f}"
                    if min(info['depths']) == max(info['depths']):
                        depth_range = f"{min(info['depths']):.3f}"

                    # Find the largest object (anchor) in this step
                    largest_obj = max(info['objects'], key=lambda x: x['area'])

                    print(f"\n  Step {step_num:2d}: {len(info['objects']):3d} objects")
                    print(f"           Depth range: {depth_range}")
                    print(f"           Total area: {info['total_area']*100:.1f}%")
                    print(f"           Anchor: {largest_obj['area_ratio']*100:.1f}% object at depth {largest_obj['median_depth']:.3f}")

                # Drawing order preview with step info
                print("\n✏️ Drawing Order Preview (First 10 objects):")
                for i in range(min(10, len(depth_ordered_objects))):
                    obj = depth_ordered_objects[i]
                    print(f"  #{obj['draw_order']:3d} [Step {obj['step']:2d}]: "
                          f"Depth={obj['median_depth']:.3f} "
                          f"({obj['layer']:10s}) "
                          f"Size={obj['area_ratio']*100:4.1f}%")

                # Depth statistics
                print("\n📊 Depth Statistics:")
                all_depths = [obj['median_depth'] for obj in depth_ordered_objects]
                print(f"  Min depth:  {min(all_depths):.3f}")
                print(f"  Max depth:  {max(all_depths):.3f}")
                print(f"  Mean depth: {np.mean(all_depths):.3f}")
                print(f"  Std depth:  {np.std(all_depths):.3f}")

            # Export ordering data
            if export_order_checkbox.value:
                # Save ordering data to file
                import json

                export_data = []
                for obj in depth_ordered_objects:
                    export_data.append({
                        'draw_order': obj['draw_order'],
                        'step': obj['step'],
                        'median_depth': float(obj['median_depth']),
                        'mean_depth': float(obj['mean_depth']),
                        'layer': obj['layer'],
                        'area_ratio': float(obj['area_ratio']),
                        'centroid': [float(obj['centroid'][0]), float(obj['centroid'][1])]
                    })

                with open('depth_ordering.json', 'w') as f:
                    json.dump(export_data, f, indent=2)

                print(f"\n✅ Ordering data exported to 'depth_ordering.json'")

                # Also save visualization
                fig.savefig('depth_ordering_visualization.png', dpi=150, bbox_inches='tight')
                print(f"✅ Visualization exported to 'depth_ordering_visualization.png'")

            # Store in pipeline results
            pipeline_results['depth_ordered_objects'] = depth_ordered_objects

            status_text.value = f'<b style="color: green;">✅ Depth ordering complete! {len(depth_ordered_objects)} objects analyzed</b>'

            print(f"\n✅ Depth-based ordering analysis complete!")
            print(f"   Objects are now ordered for drawing (background to foreground)")
            print(f"   Results stored in 'depth_ordered_objects' variable")

        except Exception as e:
            status_text.value = f'<b style="color: red;">❌ Error: {str(e)}</b>'
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()

# Connect button to handler
run_button.on_click(on_run_ordering)

# Layout
controls_box = widgets.VBox([
    widgets.HTML('<h3>📊 Depth-Based Object Ordering</h3>'),
    num_display_slider,
    depth_threshold_slider,
    min_size_slider,
    widgets.HBox([show_stats_checkbox, export_order_checkbox]),
    run_button,
    status_text
], layout=widgets.Layout(padding='10px', border='2px solid #ccc', border_radius='10px'))

display(controls_box)
display(output)

print("🎨 Depth Ordering Widget Ready!")
print("="*60)
print("This widget will:")
print("  • Group objects into meaningful depth steps")
print("  • Use large objects as anchors for each step")
print("  • Merge nearby small objects into anchor steps")
print("  • Create fewer, more coherent painting layers")
print("  • Export ordering data for artistic rendering")
print("\n⚠️ Prerequisites:")
print("  1. Run SAM 2.1 detection (Step 1)")
print("  2. Run DepthAnything V2 (Step 5)")
print("\n💡 Clustering Logic:")
print("  • Large objects (>min size) anchor depth steps")
print("  • Small objects merge into nearest anchor's step")
print("  • Results in 3-10 coherent layers typically")
print("  • Each step = paint all objects together")