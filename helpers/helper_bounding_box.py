import cv2
import os
import numpy as np

def line_intersection(p1, p2, p3, p4):
    """
    Returns the intersection point (x, y) of lines (p1,p2) and (p3,p4),
    or None if they are parallel or coincident.

    Each point should be [x, y].
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        # Parallel or coincident
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return [px, py]

def compute_boundary_lines(contour, vp, line_length):
    """
    Compute two boundary lines from a vanishing point that encompass all contour points.
    
    Args:
        contour: Contour points as numpy array of shape (N, 2) or (N, 1, 2)
        vp: Vanishing point as numpy array [x, y]
        line_length: Length of the ray lines to draw
    
    Returns:
        ray1_end: Endpoint of first boundary line (numpy array)
        ray2_end: Endpoint of second boundary line (numpy array)
    """
    # Reshape contour to (N, 2) if needed
    contour = contour.reshape(-1, 2)
    
    # Compute angles of each contour point relative to vanishing point
    angles = np.arctan2(contour[:, 1] - vp[1], contour[:, 0] - vp[0])
    
    # Find extreme angles (min and max)
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    
    # Get the corresponding contour points
    p_min = contour[min_idx]
    p_max = contour[max_idx]
    
    # Create rays for the extreme lines
    # Handle edge case where contour point is at vanishing point
    dir_min = p_min - vp
    dir_max = p_max - vp
    norm_min = np.linalg.norm(dir_min)
    norm_max = np.linalg.norm(dir_max)
    
    if norm_min > 1e-6:  # Avoid division by zero
        ray1_end = vp + dir_min / norm_min * line_length
    else:
        ray1_end = vp + np.array([1, 0]) * line_length  # Default direction
    
    if norm_max > 1e-6:  # Avoid division by zero
        ray2_end = vp + dir_max / norm_max * line_length
    else:
        ray2_end = vp + np.array([0, 1]) * line_length  # Default direction
    
    return ray1_end, ray2_end

def draw_bounding_box_on_frame(frame_image, frame_contours, vp_center_x, vp_center_y, vp_left_x, vp_left_y, vp_right_x, vp_right_y, w, h):
    """
    Draw 3D bounding boxes on a single frame.
    
    Args:
        frame_image: The frame image (numpy array)
        frame_contours: Contours for this frame
        vp_center_x, vp_center_y: Center vanishing point coordinates
        vp_left_x, vp_left_y: Left vanishing point coordinates
        vp_right_x, vp_right_y: Right vanishing point coordinates
        w: Image width
        h: Image height
    
    Returns:
        output_image: Frame with bounding boxes drawn
    """
    # Create a copy of the frame to draw on
    output_image = frame_image.copy()

    # Define vanishing point (using center vanishing point)
    vp = np.array([vp_center_x, vp_center_y])
    
    # Calculate line length as diagonal of image
    line_length = np.sqrt(w**2 + h**2)
    line_length_side = line_length * 4

    lines_bottom_front = []
    lines_bottom_inner = []

    if frame_contours:
        contours = [contour[1] for contour in frame_contours]
        # Draw contours with white color and 10% opacity
        overlay = output_image.copy()
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, output_image, 1, 0.8, output_image)
        
        # Add boundary lines for each contour
        for _, contour in enumerate(contours):            # Reshape contour to get first point
            contour_reshaped = contour.reshape(-1, 2) if len(contour.shape) == 3 else contour
            if len(contour_reshaped.shape) == 2 and contour_reshaped.shape[1] != 2:
                contour_reshaped = contour_reshaped.reshape(-1, 2)
            
            # Get first point of contour
            is_left = np.any(contour_reshaped[:, 0] < vp_center_x)
            is_right = np.any(contour_reshaped[:, 0] > vp_center_x)

            if is_left and is_right:
                continue
            
            # Choose horizontal vanishing point based on first point's x coordinate
            if is_left:
                vp_horizontal = np.array([vp_left_x, vp_left_y])
            else:
                vp_horizontal = np.array([vp_right_x, vp_right_y])
            
            # Compute boundary lines from center vanishing point
            ray1_end, ray2_end = compute_boundary_lines(contour, vp, line_length)

            # Compute boundary lines from horizontal vanishing point (left or right)
            ray3_end, ray4_end = compute_boundary_lines(contour, vp_horizontal, line_length_side)
            
            # Find smallest and largest x coordinates on the contour
            x_coords = contour_reshaped[:, 0]
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            
            # Find points with min and max x coordinates
            # If multiple points have the same x, pick the first one
            min_x_idx = np.nonzero(x_coords == min_x)[0][0]
            max_x_idx = np.nonzero(x_coords == max_x)[0][0]
            
            ray_5 = contour_reshaped[min_x_idx]
            ray_6 = contour_reshaped[max_x_idx]

            # Intersection points of 1&3, 1&5, 2&4, 2&5, 2&6, 3&6
            if is_left:
                center_bottom = ray1_end
                center_top = ray2_end
                side_bottom = ray4_end
                side_top = ray3_end
                vert_inside = ray_6
                vert_outside = ray_5
            else:
                center_bottom = ray2_end
                center_top = ray1_end
                side_bottom = ray3_end
                side_top = ray4_end
                vert_inside = ray_5
                vert_outside = ray_6
            
            front_bottom = line_intersection(vp, center_bottom, vp_horizontal, side_bottom)
            inner_bottom = line_intersection(vp, center_bottom, vert_inside, [vert_inside[0], -1])
            outer_bottom = line_intersection(vp_horizontal, side_bottom, vert_outside, [vert_outside[0], -1])
            inner_top = line_intersection(vp_horizontal, side_top, vert_inside, [vert_inside[0], -1])
            outer_top = line_intersection(vp, center_top, vert_outside, [vert_outside[0], -1])
            far_top = line_intersection(vp, center_top, vp_horizontal, side_top)
            front_top = line_intersection(vp, inner_top, front_bottom, [front_bottom[0], -1])
            # Unused: far_bottom = line_intersection(vp, outer_bottom, far_top, [far_top[0], -1])

            lines_bottom_front.append([front_bottom, outer_bottom])
            lines_bottom_inner.append([inner_bottom, front_bottom])
            
            # Store intersection points (optional: can be used for further processing)
            intersections = [
                front_bottom,
                inner_bottom,
                outer_bottom,
                inner_top,
                outer_top,
                far_top,
                front_top,
            ]
            
            # Draw intersection points (optional visualization)
            for point in intersections:
                if point is not None:
                    cv2.circle(output_image, (int(point[0]), int(point[1])), 3, (0, 0, 0), -1)
                
            # Draw line segments, a 3D rectangle around the object (b, g, r)
            cv2.line(output_image, 
                (int(front_bottom[0]), int(front_bottom[1])), 
                (int(inner_bottom[0]), int(inner_bottom[1])), 
                (0, 0, 255), 2, cv2.LINE_AA)  # Red line
            cv2.line(output_image, 
                (int(front_top[0]), int(front_top[1])), 
                (int(inner_top[0]), int(inner_top[1])), 
                (0, 0, 255), 2, cv2.LINE_AA)  # Red line
            cv2.line(output_image, 
                (int(outer_top[0]), int(outer_top[1])), 
                (int(far_top[0]), int(far_top[1])), 
                (0, 0, 255), 2, cv2.LINE_AA)  # Red line

            cv2.line(output_image, 
                (int(inner_top[0]), int(inner_top[1])), 
                (int(far_top[0]), int(far_top[1])), 
                (0, 255, 0), 2, cv2.LINE_AA)  # Green line
            cv2.line(output_image, 
                (int(front_top[0]), int(front_top[1])), 
                (int(outer_top[0]), int(outer_top[1])), 
                (0, 255, 0), 2, cv2.LINE_AA)  # Green line
            cv2.line(output_image, 
                (int(front_bottom[0]), int(front_bottom[1])), 
                (int(outer_bottom[0]), int(outer_bottom[1])), 
                (0, 255, 0), 2, cv2.LINE_AA)  # Green line

            cv2.line(output_image, 
                (int(inner_bottom[0]), int(inner_bottom[1])), 
                (int(inner_top[0]), int(inner_top[1])), 
                (255, 0, 0), 2, cv2.LINE_AA)  # Blue line
            cv2.line(output_image, 
                (int(front_bottom[0]), int(front_bottom[1])), 
                (int(front_top[0]), int(front_top[1])), 
                (255, 0, 0), 2, cv2.LINE_AA)  # Blue line
            cv2.line(output_image, 
                (int(outer_bottom[0]), int(outer_bottom[1])), 
                (int(outer_top[0]), int(outer_top[1])), 
                (255, 0, 0), 2, cv2.LINE_AA)  # Blue line

            # Draw boundary lines from center vanishing point
            # cv2.line(output_image, 
            #         (int(vp[0]), int(vp[1])), 
            #         (int(center_bottom[0]), int(center_bottom[1])), 
            #         (100, 0, 0), 2, cv2.LINE_AA)  # Red line
            # cv2.line(output_image, 
            #         (int(vp[0]), int(vp[1])), 
            #         (int(center_top[0]), int(ray2_end[1])), 
            #         (0, 0, 100), 2, cv2.LINE_AA)  # Blue line
            
            # # Draw boundary lines from horizontal vanishing point
            # cv2.line(output_image, 
            #         (int(vp_horizontal[0]), int(vp_horizontal[1])), 
            #         (int(side_bottom[0]), int(side_bottom[1])), 
            #         (0, 100, 0), 2, cv2.LINE_AA)  # Green line (ray3)
            # cv2.line(output_image, 
            #         (int(vp_horizontal[0]), int(vp_horizontal[1])), 
            #         (int(side_top[0]), int(side_top[1])), 
            #         (0, 100, 100), 2, cv2.LINE_AA)  # Yellow line (ray4)
            
            # # Draw vertical rays through the smallest and largest x coordinates
            # cv2.line(output_image,
            #         (int(vert_inside[0]), 0),
            #         (int(vert_inside[0]), h),
            #         (100, 0, 100), 2, cv2.LINE_AA)  # Magenta line for min x
            # cv2.line(output_image,
            #         (int(vert_outside[0]), 0),
            #         (int(vert_outside[0]), h),
            #         (100, 100, 0), 2, cv2.LINE_AA)  # Cyan line for max x            
    
    return output_image, lines_bottom_front, lines_bottom_inner

def find_3d_bounding_boxes(contours, vp_center_x, vp_center_y, vp_left_x, vp_left_y, vp_right_x, vp_right_y, output_path, w, h, fps=25.0):
    """
    Export a video of all frames with 3D bounding boxes drawn on contours.
    
    Args:
        contours: List of contours per frame, where each entry is (frame_image, frame_contours)
        vp_center_x, vp_center_y: Center vanishing point coordinates
        vp_left_x, vp_left_y: Left vanishing point coordinates
        vp_right_x, vp_right_y: Right vanishing point coordinates
        output_path: Path to save the exported video
        w: Image width
        h: Image height
        fps: Frames per second for the output video (default: 25.0)
    """
    if not contours or len(contours) == 0:
        print("[Export] No contours to process")
        return
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(output_path, "contours_3d_boxes.mp4")
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    
    if not out.isOpened():
        raise ValueError(f"Could not create video writer for {output_file}")
    
    # Process all frames
    lines_bottom_front_all = []
    lines_bottom_inner_all = []

    print("9 [3D bounding boxes] With {len(contours)} frames")
    for frame_idx, frame_data in enumerate(contours):
        # Handle both old format (2 elements) and new format (3 elements with object IDs)
        frame_image, frame_contours = frame_data
        
        # Draw bounding boxes on this frame
        output_frame, lines_bottom_front, lines_bottom_inner = draw_bounding_box_on_frame(
            frame_image, frame_contours,
            vp_center_x, vp_center_y, 
            vp_left_x, vp_left_y, 
            vp_right_x, vp_right_y, 
            w, h
        )

        # Collect measurement lines
        lines_bottom_front_all.extend(lines_bottom_front)
        lines_bottom_inner_all.extend(lines_bottom_inner)

        # Write frame to video
        out.write(output_frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"- Processed {frame_idx + 1}/{len(contours)} frames ({frame_idx/len(contours)*100:.1f}%)")
    
    # Release video writer
    out.release()

    return lines_bottom_front_all, lines_bottom_inner_all

