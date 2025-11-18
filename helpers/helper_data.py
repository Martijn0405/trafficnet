
import cv2
import numpy as np
import os
from helpers.helper_horizon import find_vanishing_point_horizon_frame

def data_collection(cap, model, model_seg, tracker, out, total_frames, output_path):
    contours = []
    tracking_data = []
    lines_up = []
    lines_down = []
    frame_count = 0
    process = True
        
    while process:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        results_seg = model_seg(frame, verbose=False, retina_masks=True)

        # Labels: labels = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

        # Extract detections with confidence threshold
        detections = []
        confidence_threshold = 0.5
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = boxes.cls[i].cpu().numpy()
                
                # Only keep detections above confidence threshold
                if conf >= confidence_threshold:
                    detections.append([x1, y1, x2, y2, conf, cls])

        # Update tracker first to get object IDs
        tracked_objects = tracker.update(detections)
        
        # Draw annotations and collect data
        annotated_frame = frame.copy()
        boundary_boxes = []
        
        for object_id, obj_data in tracked_objects.items():
            x1, y1, x2, y2 = obj_data['bbox']
            class_id = obj_data['class_id']
            confidence = obj_data['confidence']
            centroid = obj_data['centroid']

            # Add boundary box to list
            boundary_boxes.append([x1, y1, x2, y2])
            
            # Get class name
            class_name = model.names[class_id] if class_id in model.names else f"Class_{class_id}"
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"ID:{object_id} {class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw centroid
            cv2.circle(annotated_frame, (int(centroid[0]), int(centroid[1])), 4, color, -1)
            
            # Collect tracking data
            tracking_data.append({
                'frame_number': frame_count,
                'object_id': object_id,
                'class_name': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2,
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'bbox_width': x2 - x1,
                'bbox_height': y2 - y1,
                'bbox_area': (x2 - x1) * (y2 - y1)
            })

        # Extract masks
        contours_frame = []
        frame_seg = None
        if len(results_seg) > 0 and results_seg[0].masks is not None:
            frame_seg = np.copy(results_seg[0].orig_img)
            masks = results_seg[0].masks.data
            
            # Get frame dimensions
            h, w = frame.shape[:2]

            for i in range(len(masks)):
                # Get mask data at full resolution
                mask = masks[i].cpu().numpy()
                # Resize mask to original frame size if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Convert to binary format
                if mask.max() <= 1.0:
                    mask_binary = (mask * 255).astype(np.uint8)
                else:
                    mask_binary = mask.astype(np.uint8)
                
                # Threshold to get binary mask
                _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)
                
                # Convert mask to points (all non-zero pixel coordinates)
                mask_contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Merge all contours from this mask: map to points then append arrays
                if len(mask_contours) > 0:
                    # Map each contour to points and collect them
                    all_points = []
                    for contour in mask_contours:
                        # Reshape contour to get points
                        contour_reshaped = contour.reshape(-1, 2) if len(contour.shape) == 3 else contour
                        if len(contour_reshaped.shape) == 2 and contour_reshaped.shape[1] != 2:
                            contour_reshaped = contour_reshaped.reshape(-1, 2)
                        # Append points from this contour
                        if len(contour_reshaped) > 0:
                            all_points.append(contour_reshaped)
                    
                    # Merge all point arrays into a single array
                    if len(all_points) > 0:
                        merged_points = np.vstack(all_points)
                        contours_frame.append([i, merged_points])
                    
        contours.append([frame_seg, contours_frame])

        # Horizontal lines
        line_up, line_down = find_vanishing_point_horizon_frame(frame, boundary_boxes)
        lines_up.append(line_up)
        lines_down.append(line_down) 

        # Append lines to frame
        for (x1, y1, x2, y2) in line_up:
            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)
        for (x1, y1, x2, y2) in line_down:
            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 10 == 0:
            print(f"- Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Export frame with lines, boxes and contours
        export_enabled = True
        if output_path and export_enabled:
            # Create contours folder if it doesn't exist
            contours_dir = os.path.join(output_path, "contours")
            os.makedirs(contours_dir, exist_ok=True)
            
            # Create a copy of the frame for export
            export_frame = frame.copy()
            
            # Draw boundary boxes
            for bbox in boundary_boxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(export_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw lines_up (blue)
            for (x1, y1, x2, y2) in line_up:
                cv2.line(export_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)
            
            # Draw lines_down (red)
            for (x1, y1, x2, y2) in line_down:
                cv2.line(export_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw contours (now merged points)
            if contours_frame:
                for _, contour_data in enumerate(contours_frame):
                    mask_index = contour_data[0]  # The mask index i
                    merged_points = contour_data[1]  # The merged points array
                    
                    # Ensure points are in correct format (N, 2)
                    merged_points = merged_points.reshape(-1, 2)
                    
                    if len(merged_points) > 0:
                        # Draw points as circles
                        for point in merged_points:
                            x, y = int(point[0]), int(point[1])
                            if 0 <= x < w and 0 <= y < h:
                                cv2.circle(export_frame, (x, y), 1, (255, 255, 0), -1)  # Cyan color for points
                        
                        # Find top-left point (minimum x and y) for text placement
                        top_left_idx = np.argmin(merged_points[:, 0] + merged_points[:, 1])
                        text_x = int(merged_points[top_left_idx, 0])
                        text_y = int(merged_points[top_left_idx, 1]) - 5
                        
                        # Ensure text is within image bounds
                        text_x = max(5, min(text_x, w - 80))
                        text_y = max(20, min(text_y, h - 5))
                        
                        # Draw text background
                        label = f"ID: {mask_index}"
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(export_frame, 
                                     (text_x - 3, text_y - text_height - 3), 
                                     (text_x + text_width + 3, text_y + baseline + 3), 
                                     (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(export_frame, label, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save frame as {frame_count}.png
            frame_filename = os.path.join(contours_dir, f"{frame_count}.png")
            cv2.imwrite(frame_filename, export_frame)
        
        # Stop if frame_count is greater than total_frames
        if frame_count >= total_frames:
            process = False

    # Flatten lines_up and lines_down
    lines_up = [item for sublist in lines_up for item in sublist]
    lines_down = [item for sublist in lines_down for item in sublist]

    return contours, tracking_data, lines_up, lines_down