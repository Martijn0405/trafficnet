import matplotlib.pyplot as plt
import math
import cv2
import numpy as np

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
        Tuple of 4 points: (point_bl, point_br, point_tl, point_tr)
        - Point bl: Leftmost intersection at y = y_centroid_min (lowest x)
        - Point br: Rightmost intersection at y = y_centroid_min (highest x)
        - Point tl: On line from Point bl to vp_center at y = y_centroid_max
        - Point tr: On line from Point br to vp_center at y = y_centroid_max
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
    point_bl = min(intersection_points, key=lambda p: p[0])
    point_bl_x, point_bl_y = point_bl
    
    # Point 2: Rightmost intersection (highest x)
    point_br = max(intersection_points, key=lambda p: p[0])
    point_br_x, point_br_y = point_br
    
    # Point 3: On line from Point 1 to vp_center at y = y_centroid_max
    # Line from (point_bl_x, y_centroid_min) to (vp_center_x, vp_center_y)
    # Using parametric: x = point_bl_x + t*(vp_center_x - point_bl_x)
    #                   y = y_centroid_min + t*(vp_center_y - y_centroid_min)
    # Solve for t when y = y_centroid_max
    if abs(vp_center_y - y_centroid_min) < 1e-6:
        # Line is horizontal, cannot find point at y_centroid_max
        point_tl_x = point_bl_x
        point_tl_y = y_centroid_min
    else:
        t3 = (y_centroid_max - y_centroid_min) / (vp_center_y - y_centroid_min)
        point_tl_x = point_bl_x + t3 * (vp_center_x - point_bl_x)
        point_tl_y = y_centroid_max
    
    point_tl = (point_tl_x, point_tl_y)
    
    # Point tr: On line from Point br to vp_center at y = y_centroid_max
    # Line from (point_br_x, y_centroid_min) to (vp_center_x, vp_center_y)
    if abs(vp_center_y - y_centroid_min) < 1e-6:
        # Line is horizontal, cannot find point at y_centroid_max
        point_tr_x = point_br_x
        point_tr_y = y_centroid_min
    else:
        t4 = (y_centroid_max - y_centroid_min) / (vp_center_y - y_centroid_min)
        point_tr_x = point_br_x + t4 * (vp_center_x - point_br_x)
        point_tr_y = y_centroid_max
    
    point_tr = (point_tr_x, point_tr_y)
    
    # Create visualization
    if output_path:
        _, ax = plt.subplots(figsize=(w/100, h/100))
        
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
        ax.scatter([point_bl_x], [point_bl_y], color='green', s=150, marker='o', 
                  edgecolors='darkgreen', linewidths=3, zorder=6, label='Point 1 (leftmost)')
        ax.scatter([point_br_x], [point_br_y], color='blue', s=150, marker='s', 
                  edgecolors='darkblue', linewidths=3, zorder=6, label='Point 2 (rightmost)')
        ax.scatter([point_tl_x], [point_tl_y], color='purple', s=150, marker='^', 
                  edgecolors='darkviolet', linewidths=3, zorder=6, label='Point 3')
        ax.scatter([point_tr_x], [point_tr_y], color='cyan', s=150, marker='v', 
                  edgecolors='darkcyan', linewidths=3, zorder=6, label='Point 4')
        
        # Draw 4 lines connecting the points to form a quadrilateral
        ax.plot([point_bl_x, point_br_x], [point_bl_y, point_br_y], 'black', linewidth=3, alpha=0.8, zorder=5, label='Quadrilateral edge')
        ax.plot([point_br_x, point_tr_x], [point_br_y, point_tr_y], 'black', linewidth=3, alpha=0.8, zorder=5)
        ax.plot([point_tr_x, point_tl_x], [point_tr_y, point_tl_y], 'black', linewidth=3, alpha=0.8, zorder=5)
        ax.plot([point_tl_x, point_bl_x], [point_tl_y, point_bl_y], 'black', linewidth=3, alpha=0.8, zorder=5)
        
        # Draw lines from points 1 and 2 to vanishing point
        ax.plot([point_bl_x, vp_center_x], [point_bl_y, vp_center_y], 'g--', linewidth=2, alpha=0.7, label='Line to VP from Point 1')
        ax.plot([point_br_x, vp_center_x], [point_br_y, vp_center_y], 'b--', linewidth=2, alpha=0.7, label='Line to VP from Point 2')
        
        # Draw vanishing point
        ax.scatter([vp_center_x], [vp_center_y], color='yellow', s=200, marker='*', 
                  edgecolors='black', linewidths=2, zorder=7, label='Vanishing Point')
        
        # Set labels and limits
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_title('Hough Lines Intersections - 4 Key Points', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.invert_yaxis()  # Match image coordinates (origin at top-left)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_path + "/hough_lines_bottom_intersections.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return point_bl, point_br, point_tl, point_tr

def compute_road_length_fov(h, fov, road_width_px, road_width_m, vp_center_y, point_bl, point_tl):
    # FOV formula: f = (H/2) / tan(FOV/2)
    f = (h/2) / math.tan(math.radians(fov/2))

    # Z_relative = 1 / (y_pixel - horizon_pixel)
    # z_near_rel = 1 / (point_bl[1] - vp_center_y)
    # z_far_rel = 1 / (point_tl[1] - vp_center_y)

    # Geometric ratio
    # L_rel = z_far_rel - z_near_rel
    est_z_near = (road_width_m * f) / road_width_px
    depth_ratio = (point_bl[1] - vp_center_y) / (point_tl[1] - vp_center_y)
    est_z_far = est_z_near * depth_ratio

    # Physical length of road segment
    road_length_m = est_z_far - est_z_near

    return road_length_m

def bev_conversion(road_width_m, road_length_m, src_pts):
    # Mapped points in metric space
    dst_metric_pts = np.float32([
        [0, road_length_m],                        # Bottom-Left (Near Left)
        [road_width_m, road_length_m],             # Bottom-Right (Near Right)
        [0, 0],            # Top-Left (Far Left)
        [road_width_m, 0], # Top-Right (Far Right)
    ])

    # H
    h = cv2.getPerspectiveTransform(src_pts, dst_metric_pts)
    h_inv = np.linalg.inv(h)

    def bev_px_to_world(x, y):
        # Create a vector [x, y, 1]
        pt_vec = np.array([[[x, y]]], dtype=np.float32)

        # Transform
        dst_vec = cv2.perspectiveTransform(pt_vec, h)

        # Extract x, y
        x_meters = dst_vec[0][0][0]
        y_meters = dst_vec[0][0][1]
        
        return x_meters, y_meters
    
    def bev_world_to_px(x_m, y_m):
        # Create a vector [x, y, 1]
        pt_vec = np.array([[[x_m, y_m]]], dtype=np.float32)
        dst_vec = cv2.perspectiveTransform(pt_vec, h_inv)
        
        u_px = int(dst_vec[0][0][0])
        v_px = int(dst_vec[0][0][1])
        
        return u_px, v_px
    
    return bev_px_to_world, bev_world_to_px

def build_centroids_per_vehicle_id_bev(road_width_m, road_length_m, centroids_per_vehicle_id, bev_px_to_world, output_path):
    """
    Convert centroids to BEV coordinates (meters) and export visualization.
    
    Args:
        road_width_m: Width of the road in meters
        road_length_m: Length of the road in meters
        centroids_per_vehicle_id: Dictionary of vehicle centroids
        bev_px_to_world: Function to convert pixel coordinates to world coordinates
        output_path: Path to save the exported PNG
    
    Returns:
        centroids_bev: Dictionary of vehicle centroids in world coordinates
    """
    # Create a blank image for BEV
    # Scale: 20 pixels per meter
    scale = 20
    img_w = int(road_width_m * scale)
    img_h = int(road_length_m * scale)
    
    # Ensure valid image dimensions
    img_w = max(1, img_w)
    img_h = max(1, img_h)
    
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    centroids_bev = {}
    
    # Generate colors (same as helper_centroids.py)
    num_vehicles = len(centroids_per_vehicle_id)
    colors = []
    if num_vehicles > 0:
        hue_step = 180 // max(num_vehicles, 1)
        for i in range(num_vehicles):
            hue = i * hue_step
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])))

    for idx, (vehicle_id, centroids) in enumerate(centroids_per_vehicle_id.items()):
        if len(centroids) == 0:
            continue
            
        color = colors[idx % len(colors)] if colors else (0, 255, 0)
        
        centroids_bev[vehicle_id] = []
        points = []
        
        for centroid in centroids:
            # Convert px to world
            cx, cy = centroid['centroid_x'], centroid['centroid_y']
            wx, wy = bev_px_to_world(cx, cy)

            print(f"- Vehicle {vehicle_id}: {cx}, {cy}, {wx}, {wy}")
            
            centroids_bev[vehicle_id].append({
                'frame_number': centroid['frame_number'],
                'x_m': wx,
                'y_m': wy
            })
            
            # Map to image coordinates
            # wx ranges from 0 to road_width_m
            # wy ranges from 0 to road_length_m
            
            # Transform to image coordinates (origin top-left)
            # y increases downwards in image, but increases "upwards" in world (away from camera)
            # So wy=0 is bottom of image (img_h), wy=road_length_m is top (0)
            
            px = int(wx * scale)
            py = int(img_h - (wy * scale))
            
            # Allow drawing slightly outside for continuity, but clip for points
            points.append((px, py))

        # Draw trajectory
        if len(points) > 1:
            for i in range(len(points) - 1):
                # Clip lines to image bounds for safety (cv2 usually handles this but to be safe)
                p1 = points[i]
                p2 = points[i+1]
                cv2.line(img, p1, p2, color, 2)
        
        # Draw points
        for pt in points:
             # Only draw points if they are within the image
             if 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
                 cv2.circle(img, pt, 3, color, -1)
                 cv2.circle(img, pt, 5, color, 1)

    if output_path:
        output_file = f"{output_path}/centroids_bev.png"
        cv2.imwrite(output_file, img)

    return centroids_bev