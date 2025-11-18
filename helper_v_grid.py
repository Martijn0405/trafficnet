import numpy as np
import cv2

def draw_vanishing_grid(output_path,image, w, h, vp_center_x, vp_center_y, vp_left_x, vp_left_y, vp_right_x, vp_right_y):
    image = image.copy()

    grid_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for x_percentage in grid_percentages:
        for y_percentage in grid_percentages:
            start_x = int(w * x_percentage)
            start_y = int(h * y_percentage)

            # Center VP
            dx = vp_center_x - start_x
            dy = vp_center_y - start_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = (dx / length) * 40
                dy_norm = (dy / length) * 40
                cv2.arrowedLine(image, (start_x, start_y), (int(start_x + dx_norm), int(start_y + dy_norm)), (0, 0, 255), 2, tipLength=0.3)
            
            # Left VP
            dx = vp_left_x - start_x
            dy = vp_left_y - start_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = (dx / length) * 40
                dy_norm = (dy / length) * 40
                cv2.arrowedLine(image, (start_x, start_y), (int(start_x + dx_norm), int(start_y + dy_norm)), (255, 0, 0), 2, tipLength=0.3)
            
            # Right VP
            dx = vp_right_x - start_x
            dy = vp_right_y - start_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = (dx / length) * 40
                dy_norm = (dy / length) * 40
                cv2.arrowedLine(image, (start_x, start_y), (int(start_x + dx_norm), int(start_y + dy_norm)), (0, 255, 0), 2, tipLength=0.3)

    cv2.circle(image, (int(vp_center_x), int(vp_center_y)), 8, (0, 0, 255), -1)
    cv2.circle(image, (int(vp_center_x), int(vp_center_y)), 12, (255, 255, 255), 2)
    cv2.circle(image, (int(vp_left_x), int(vp_left_y)), 8, (0, 0, 255), -1)
    cv2.circle(image, (int(vp_left_x), int(vp_left_y)), 12, (255, 255, 255), 2)
    cv2.circle(image, (int(vp_right_x), int(vp_right_y)), 8, (0, 0, 255), -1)
    cv2.circle(image, (int(vp_right_x), int(vp_right_y)), 12, (255, 255, 255), 2)

    grid_path = output_path + "/vanishing_grid.jpg"
    cv2.imwrite(grid_path, image)
