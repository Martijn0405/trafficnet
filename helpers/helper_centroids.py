import cv2
import numpy as np

def build_centroids_per_vehicle_id(tracking_data, output_path, w, h):
    centroids_per_vehicle_id = {}
    for detection in tracking_data:
        object_id = detection['object_id']
        if object_id not in centroids_per_vehicle_id:
            centroids_per_vehicle_id[object_id] = []
        centroids_per_vehicle_id[object_id].append({
            'frame_number': detection['frame_number'],
            'centroid_x': detection['centroid_x'],
            'centroid_y': detection['centroid_y']
        })
    for vehicle_id in centroids_per_vehicle_id:
        centroids_per_vehicle_id[vehicle_id].sort(key=lambda x: x['frame_number'])

    # Export points on a w by h image
    if output_path:
        # Create a blank image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate colors for each vehicle
        num_vehicles = len(centroids_per_vehicle_id)
        colors = []
        if num_vehicles > 0:
            hue_step = 180 // max(num_vehicles, 1)  # Use HSV color space for better color distribution
            for i in range(num_vehicles):
                hue = i * hue_step
                color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                colors.append((int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])))
        
        # Draw points and trajectories for each vehicle
        for idx, (vehicle_id, centroids) in enumerate(centroids_per_vehicle_id.items()):
            if len(centroids) == 0:
                continue
            
            color = colors[idx % len(colors)] if colors else (0, 255, 0)
            
            # Draw trajectory lines
            if len(centroids) > 1:
                points = []
                for centroid in centroids:
                    x = int(centroid['centroid_x'])
                    y = int(centroid['centroid_y'])
                    points.append((x, y))
                
                # Draw lines connecting consecutive points
                for i in range(len(points) - 1):
                    cv2.line(img, points[i], points[i + 1], color, 2)
            
            # Draw points
            for centroid in centroids:
                x = int(centroid['centroid_x'])
                y = int(centroid['centroid_y'])
                # Draw circle for each point
                cv2.circle(img, (x, y), 3, color, -1)
                # Draw a smaller circle for better visibility
                cv2.circle(img, (x, y), 5, color, 1)
        
        # Save the image
        output_file = f"{output_path}/centroids_points.png"
        cv2.imwrite(output_file, img)

    return centroids_per_vehicle_id