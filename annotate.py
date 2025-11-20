import cv2
from ultralytics import YOLO
import os
import numpy as np

from helpers.helper_v_point import diamond_accumulator, find_vanishing_lines_center, find_vanishing_point_center
from helpers.helper_v_grid import draw_vanishing_grid
from helpers.helper_centroids import build_centroids_per_vehicle_id
from helpers.helper_trajectories import export_trajectories
from helpers.helper_tracker import ObjectTracker
from helpers.helper_horizon import find_vanishing_point_horizon
from helpers.helper_bev import bev_conversion, find_hough_lines_bottom_intersections, build_centroids_per_vehicle_id_bev
from helpers.helper_bounding_box import find_3d_bounding_boxes
from helpers.helper_data import data_collection
from helpers.helper_size import compare_line_depths_to_function, compute_lines_distance, compute_trapezoid_width_function, compare_line_widths_to_function, compute_scene_depth
   
def main(input_path="/input/video.mp4", output_path="export", model_name="/models/yolo11s.pt", model_seg_name="/models/yolo11s-seg.pt", fov=45):
    print("1 [Process] Start")

    # Model
    print("2 [Model] Loading")
    model = YOLO(model_name)
    model_seg = YOLO(model_seg_name)
    
    # Video loading
    print("3 [Video] Loading")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"- Video is {w}x{h}, {fps} FPS, {total_frames} frames")
    
    # Video processing
    print("4 [Process] Video processing")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path + "/annotated.mp4", fourcc, fps, (w, h))
    tracker = ObjectTracker(max_disappeared=30, max_distance=100)

    # At most 50 frames
    process_count = min(total_frames, 100)
    
    # Data collection
    contours, tracking_data, lines_up, lines_down = data_collection(cap, model, model_seg, tracker, out, process_count, output_path)

    # Trajectories
    print("5 [VP] Computing vanishing point")
    centroids_per_vehicle_id = build_centroids_per_vehicle_id(tracking_data, output_path, w, h)
    all_lines = find_vanishing_lines_center(centroids_per_vehicle_id)
    all_lines_flat = [item for sublist in all_lines for item in sublist]
    if len(all_lines_flat) == 0:
        print("[Process] Stopped, All Lines Flat")
        return

    # Find legible vehicles
    legible_vehicles = []
    legible_lines = []
    for i, lines in enumerate(all_lines):
        if len(lines) >= 10:
            legible_vehicles.append(i)
            legible_lines.append(lines)

    hough_lines = []
    for i, legible_line in enumerate(legible_lines):
        hough_line = diamond_accumulator(legible_line, w, h, output_path, i+1)
        hough_lines.extend(hough_line)
    
    # Center vanishing point
    vp_center_x, vp_center_y = find_vanishing_point_center(all_lines_flat, hough_lines, output_path, w, h)
    if vp_center_x is None or vp_center_y is None:
        print("[Process] Stopped, VP Center")
        return
    export_trajectories(centroids_per_vehicle_id, output_path, w, h)
    
    # Horizontal vanishing points
    vp_left_x, vp_left_y, vp_right_x, vp_right_y = find_vanishing_point_horizon(lines_up, lines_down, vp_center_x,vp_center_y, output_path, w, h)

    # Map centroids to bird's eye view
    print("6 [BEV] Mapping centroids to bird's eye view")
    y_centroid_min = max(centroid['centroid_y'] for centroids in centroids_per_vehicle_id.values() for centroid in centroids)
    y_centroid_max = min(centroid['centroid_y'] for centroids in centroids_per_vehicle_id.values() for centroid in centroids)
    point_bl, point_br, point_tl, point_tr = find_hough_lines_bottom_intersections(vp_center_x, vp_center_y, y_centroid_min, y_centroid_max, hough_lines, w, h, output_path)
    print(f"- Point 1 (bl): ({point_bl[0]:.2f}, {point_bl[1]:.2f})")
    print(f"- Point 2 (br): ({point_br[0]:.2f}, {point_br[1]:.2f})")
    print(f"- Point 3 (tl): ({point_tl[0]:.2f}, {point_tl[1]:.2f})")
    print(f"- Point 4 (tr): ({point_tr[0]:.2f}, {point_tr[1]:.2f})")

    # Get first frame for vanishing grid before releasing capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
    ret, first_frame = cap.read()

    # 3D bounding box
    print(f"8 [3D bounding boxes] With {len(contours)} frames")
    lines_width, lines_depth, _ = find_3d_bounding_boxes(contours, vp_center_x, vp_center_y, vp_left_x, vp_left_y, vp_right_x, vp_right_y, output_path, w, h, fps)

    # Avg car dimensions (1.51 x 4.27 x 1.74 meters)
    avg_car_width = 1.51
    avg_car_length = 4.27
    
    # Estimate width of scene
    print("9 [Size] Trapezoid width function")
    src_pts = np.float32([point_bl, point_br, point_tl, point_tr])
    width_function = compute_trapezoid_width_function(src_pts)
    lines_width_distance = compute_lines_distance(lines_width)
    average_ratio_width = compare_line_widths_to_function(lines_width_distance, width_function)
    road_width_px = width_function(point_bl[1])
    road_width_m = (1 / average_ratio_width) * avg_car_width
    print(f"- Road width: {road_width_px:.2f}px / {(road_width_m):.2f}m")

    # Estimate depth of scene
    print("10 [Size] Depth function")
    depth_function = compute_scene_depth(width_function)
    lines_depth_distance = compute_lines_distance(lines_depth)  
    average_ratio_depth = compare_line_depths_to_function(lines_depth_distance, depth_function)
    road_depth_px= depth_function(point_bl, point_tl)
    road_depth_m = (1 / average_ratio_depth) * avg_car_length
    print(f"- Road depth: {road_depth_px:.2f}px / {(road_depth_m):.2f}m")
    
    # BEV conversion
    print("11 [BEV] Estimating BEV")
    bev_px_to_world, _ = bev_conversion(road_width_m, road_depth_m, src_pts)
    build_centroids_per_vehicle_id_bev(road_width_m, road_depth_m, centroids_per_vehicle_id, bev_px_to_world, output_path)

    # Cleanup
    cap.release()
    out.release()

    # Export vanishing grid
    if ret: draw_vanishing_grid(output_path, first_frame, w, h, vp_center_x, vp_center_y, vp_left_x, vp_left_y, vp_right_x, vp_right_y)
    
    print("[Completed]")

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), "input", "video.mp4")
    output_path = os.path.join(os.path.dirname(__file__), "output")
    model_name = os.path.join(os.path.dirname(__file__), "models", "yolo11s.pt")
    model_seg_name = os.path.join(os.path.dirname(__file__), "models", "yolo11s-seg.pt")
    fov = 45

    main(input_path, output_path, model_name, model_seg_name, fov)
