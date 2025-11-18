import cv2
from ultralytics import YOLO
import os

from helpers.helper_v_point import diamond_accumulator, find_vanishing_lines_center, find_vanishing_point_center
from helpers.helper_v_grid import draw_vanishing_grid
from helpers.helper_centroids import build_centroids_per_vehicle_id
from helpers.helper_trajectories import export_trajectories
from helpers.helper_tracker import ObjectTracker
from helpers.helper_horizon import find_vanishing_point_horizon
from helpers.helper_bev import find_hough_lines_bottom_intersections
from helpers.helper_bounding_box import find_3d_bounding_boxes
from helpers.helper_data import data_collection
from helpers.helper_size import compute_trapezoid_width_function, compute_lines_width, compare_line_widths_to_function, compute_scene_depth
   
def main(input_path="/input/video.mp4", output_path="export", model_name="/models/yolo11s.pt", model_seg_name="/models/yolo11s-seg.pt"):
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
    point1, point2, point3, point4 = find_hough_lines_bottom_intersections(vp_center_x, vp_center_y, y_centroid_min, y_centroid_max, hough_lines, w, h, output_path)
    print(f"- Point 1 (bl): ({point1[0]:.2f}, {point1[1]:.2f})")
    print(f"- Point 2 (br): ({point2[0]:.2f}, {point2[1]:.2f})")
    print(f"- Point 3 (tl): ({point3[0]:.2f}, {point3[1]:.2f})")
    print(f"- Point 4 (tr): ({point4[0]:.2f}, {point4[1]:.2f})")

    # Get first frame for vanishing grid before releasing capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
    ret, first_frame = cap.read()

    # 3D bounding box
    print(f"8 [3D bounding boxes] With {len(contours)} frames")
    lines_bottom_front, lines_bottom_inner = find_3d_bounding_boxes(contours, vp_center_x, vp_center_y, vp_left_x, vp_left_y, vp_right_x, vp_right_y, output_path, w, h, fps)
    
    # Estimate width of scene
    print("9 [Size] Trapezoid width function")
    width_function = compute_trapezoid_width_function(point1, point2, point3, point4)
    lines_bottom_front_width = compute_lines_width(lines_bottom_front)
    average_ratio_width = compare_line_widths_to_function(lines_bottom_front_width, width_function)
    width_bottom_px = width_function(point1[1])
    width_bottom_cm = (1 / average_ratio_width) * 175

    print(f"- Width bottom PX: {width_bottom_px:.2f}px")
    print(f"- Width bottom M: {(width_bottom_cm/100):.2f}m")

    # Estimate depth of scene
    print("10 [Size] Depth function")
    depth_function = compute_scene_depth(0, point1, point2, vp_center_x, vp_center_y)
    print(f"- Depth function at 0%: {depth_function}")
    lines_bottom_inner_width = compute_lines_width(lines_bottom_inner)
    print(f"- Lines bottom inner width: {lines_bottom_inner_width}")

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
    main(input_path, output_path, model_name, model_seg_name)
