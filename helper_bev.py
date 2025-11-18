import numpy as np
import matplotlib.pyplot as plt

def find_hough_lines_bottom_intersections(vp_center_x, vp_center_y, y_centroid_min, y_centroid_max, hough_lines, w, h, output_path=None):
    """
    Find 4 key points based on hough_lines intersections with y_centroid_min and lines to vanishing point.
    Each line is represented as [x1, y1, x2, y2].
    
    Args:
        vp_center_x, vp_center_y: Center vanishing point coordinates
        y_centroid_min: Y coordinate for points 1 and 2 (leftmost and rightmost intersections)
        y_centroid_max: Y coordinate for points 3 and 4
        hough_lines: List of lines, each in format [x1, y1, x2, y2]
        w: Image width
        h: Image height (bottom of screen)
        output_path: Optional path to save the visualization PNG
    
    Returns:
        Tuple of 4 points: (point1, point2, point3, point4)
        - Point 1: Leftmost intersection at y = y_centroid_min (lowest x)
        - Point 2: Rightmost intersection at y = y_centroid_min (highest x)
        - Point 3: On line from Point 1 to vp_center at y = y_centroid_max
        - Point 4: On line from Point 2 to vp_center at y = y_centroid_max
    """
    intersection_points = []
    
    # Find all intersections of hough_lines with y = y_centroid_min
    for line in hough_lines:
        x1, y1, x2, y2 = line
        
        # Check if line is vertical (infinite slope)
        if abs(x2 - x1) < 1e-6:
            # Vertical line: check if y = y_centroid_min is between y1 and y2
            if min(y1, y2) <= y_centroid_min <= max(y1, y2):
                x_intersect = x1
                intersection_points.append((x_intersect, y_centroid_min))
            continue
        
        # Check if line is horizontal
        if abs(y2 - y1) < 1e-6:
            # Horizontal line: only intersects if y1 == y_centroid_min
            if abs(y1 - y_centroid_min) < 1e-6:
                # Line is exactly at y = y_centroid_min, use the x-range
                intersection_points.append((x1, y_centroid_min))
                intersection_points.append((x2, y_centroid_min))
            continue
        
        # General case: line equation y = m*x + b or parametric form
        # Using parametric: x = x1 + t*(x2-x1), y = y1 + t*(y2-y1)
        # Solve for t when y = y_centroid_min: y_centroid_min = y1 + t*(y2-y1)
        # t = (y_centroid_min - y1) / (y2 - y1)
        t = (y_centroid_min - y1) / (y2 - y1)
        
        # Calculate x coordinate at y = y_centroid_min
        x_intersect = x1 + t * (x2 - x1)
        
        # Include all intersections regardless of image bounds
        intersection_points.append((x_intersect, y_centroid_min))
    
    if not intersection_points:
        print(f"Warning: No intersection points found at y = {y_centroid_min}")
        return None, None, None, None
    
    # Point 1: Leftmost intersection (lowest x)
    point1 = min(intersection_points, key=lambda p: p[0])
    point1_x, point1_y = point1
    
    # Point 2: Rightmost intersection (highest x)
    point2 = max(intersection_points, key=lambda p: p[0])
    point2_x, point2_y = point2
    
    # Point 3: On line from Point 1 to vp_center at y = y_centroid_max
    # Line from (point1_x, y_centroid_min) to (vp_center_x, vp_center_y)
    # Using parametric: x = point1_x + t*(vp_center_x - point1_x)
    #                   y = y_centroid_min + t*(vp_center_y - y_centroid_min)
    # Solve for t when y = y_centroid_max
    if abs(vp_center_y - y_centroid_min) < 1e-6:
        # Line is horizontal, cannot find point at y_centroid_max
        point3_x = point1_x
        point3_y = y_centroid_min
    else:
        t3 = (y_centroid_max - y_centroid_min) / (vp_center_y - y_centroid_min)
        point3_x = point1_x + t3 * (vp_center_x - point1_x)
        point3_y = y_centroid_max
    
    point3 = (point3_x, point3_y)
    
    # Point 4: On line from Point 2 to vp_center at y = y_centroid_max
    # Line from (point2_x, y_centroid_min) to (vp_center_x, vp_center_y)
    if abs(vp_center_y - y_centroid_min) < 1e-6:
        # Line is horizontal, cannot find point at y_centroid_max
        point4_x = point2_x
        point4_y = y_centroid_min
    else:
        t4 = (y_centroid_max - y_centroid_min) / (vp_center_y - y_centroid_min)
        point4_x = point2_x + t4 * (vp_center_x - point2_x)
        point4_y = y_centroid_max
    
    point4 = (point4_x, point4_y)
    
    # Create visualization
    if output_path:
        fig, ax = plt.subplots(figsize=(w/100, h/100))
        
        # Draw each hough line
        for line in hough_lines:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.3)
        
        # Draw bottom line (y = h)
        ax.axhline(y=h, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Bottom of screen (y = h)')
        
        # Draw line at y_centroid_min
        ax.axhline(y=y_centroid_min, color='green', linestyle='--', linewidth=2, label=f'y = {y_centroid_min} (centroid min)')
        
        # Draw line at y_centroid_max
        ax.axhline(y=y_centroid_max, color='orange', linestyle='--', linewidth=2, label=f'y = {y_centroid_max} (centroid max)')
        
        # Draw all intersection points at y_centroid_min
        if intersection_points:
            x_points = [p[0] for p in intersection_points]
            y_points = [p[1] for p in intersection_points]
            ax.scatter(x_points, y_points, color='red', s=50, marker='o', 
                      edgecolors='darkred', linewidths=1, zorder=4, alpha=0.5,
                      label=f'All intersections ({len(intersection_points)})')
        
        # Draw the 4 key points
        ax.scatter([point1_x], [point1_y], color='green', s=150, marker='o', 
                  edgecolors='darkgreen', linewidths=3, zorder=6, label='Point 1 (leftmost)')
        ax.scatter([point2_x], [point2_y], color='blue', s=150, marker='s', 
                  edgecolors='darkblue', linewidths=3, zorder=6, label='Point 2 (rightmost)')
        ax.scatter([point3_x], [point3_y], color='purple', s=150, marker='^', 
                  edgecolors='darkviolet', linewidths=3, zorder=6, label='Point 3')
        ax.scatter([point4_x], [point4_y], color='cyan', s=150, marker='v', 
                  edgecolors='darkcyan', linewidths=3, zorder=6, label='Point 4')
        
        # Draw 4 lines connecting the points to form a quadrilateral
        ax.plot([point1_x, point2_x], [point1_y, point2_y], 'black', linewidth=3, alpha=0.8, zorder=5, label='Quadrilateral edge')
        ax.plot([point2_x, point4_x], [point2_y, point4_y], 'black', linewidth=3, alpha=0.8, zorder=5)
        ax.plot([point4_x, point3_x], [point4_y, point3_y], 'black', linewidth=3, alpha=0.8, zorder=5)
        ax.plot([point3_x, point1_x], [point3_y, point1_y], 'black', linewidth=3, alpha=0.8, zorder=5)
        
        # Draw lines from points 1 and 2 to vanishing point
        ax.plot([point1_x, vp_center_x], [point1_y, vp_center_y], 'g--', linewidth=2, alpha=0.7, label='Line to VP from Point 1')
        ax.plot([point2_x, vp_center_x], [point2_y, vp_center_y], 'b--', linewidth=2, alpha=0.7, label='Line to VP from Point 2')
        
        # Draw vanishing point
        ax.scatter([vp_center_x], [vp_center_y], color='yellow', s=200, marker='*', 
                  edgecolors='black', linewidths=2, zorder=7, label='Vanishing Point')
        
        # Set labels and limits
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_title(f'Hough Lines Intersections - 4 Key Points', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.invert_yaxis()  # Match image coordinates (origin at top-left)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_path + "/hough_lines_bottom_intersections.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return point1, point2, point3, point4
