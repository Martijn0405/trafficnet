import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers.helper_intersection import compute_vanishing_point
from helpers.helper_functions import is_point_in_boundary_box

def find_vanishing_point_horizon(horizontal_lines_up, horizontal_lines_down, x_value=None,y_value=None, output_path=None, w=500, h=500):
    names = ["vehicle"]

    # Filter lines_up to only include segments where x2 > x_value
    if x_value is not None:
        lines_up_filtered = [line for line in horizontal_lines_up if line[0] < x_value]
    else:
        lines_up_filtered = horizontal_lines_up

    # Right side vanishing point
    vp_right = compute_vanishing_point(lines_up_filtered, y_value)

    if output_path:
        for j, line_set in enumerate([lines_up_filtered]):
            plt.figure(figsize=(12, 8))
            for i, line in enumerate(line_set):
                x1, y1, x2, y2 = line
                color = plt.cm.viridis(i / len(line_set))
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

            # Plot vp_flat
            plt.plot(vp_right[0], vp_right[1], 'ro', markersize=10, label='Vanishing Point')
            plt.legend()

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.xlabel('X Coordinate (pixels)', fontsize=12)
            plt.ylabel('Y Coordinate (pixels)', fontsize=12)
            plt.title(f'Line Segments Used in Hough Transform ({len(line_set)} segments)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(line_set)))
            sm.set_array([])
            plt.colorbar(sm, ax=plt.gca(), label='Line Segment Index')
            plt.tight_layout()
            plt.savefig(output_path + "/vp_sides/hough_lines_" + names[j] + "_up.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Filter lines_down to only include segments where x1 < x_value
    if x_value is not None:
        lines_down_filtered = [line for line in horizontal_lines_down if line[2] > x_value]
    else:
        lines_down_filtered = horizontal_lines_down

    # Left side vanishing point
    vp_left = compute_vanishing_point(lines_down_filtered, y_value)

    if output_path:
        for j, line_set in enumerate([lines_down_filtered]):
            plt.figure(figsize=(12, 8))
            for i, line in enumerate(line_set):
                x1, y1, x2, y2 = line
                color = plt.cm.viridis(i / len(line_set))
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

            # Plot vp_flat
            plt.plot(vp_left[0], vp_left[1], 'ro', markersize=10, label='Vanishing Point')
            plt.legend()

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.xlabel('X Coordinate (pixels)', fontsize=12)
            plt.ylabel('Y Coordinate (pixels)', fontsize=12)
            plt.title(f'Line Segments Used in Hough Transform ({len(line_set)} segments)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(line_set)))
            sm.set_array([])
            plt.colorbar(sm, ax=plt.gca(), label='Line Segment Index')
            plt.tight_layout()
            plt.savefig(output_path + "/vp_sides/hough_lines_" + names[j] + "_down.png", dpi=300, bbox_inches='tight')
            plt.close()

    return vp_right[0], vp_right[1], vp_left[0], vp_left[1]

def find_vanishing_point_horizon_frame(first_frame, boundary_boxes):
    """
    Find horizontal lines (within ±10 degrees of horizontal) to detect the horizon line
    and compute the second vanishing point.
    """
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 50
    max_line_gap = 20

    hough_lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )

    horizontal_lines_up = []
    horizontal_lines_down = []
    angle_threshold = 10

    if hough_lines is not None:
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                if not is_point_in_boundary_box(x1, y1,x2, y2, boundary_boxes): continue

                dx = x2 - x1
                dy = y2 - y1
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)

                if angle_deg > 180:
                    angle_deg -= 360
                elif angle_deg < -180:
                    angle_deg += 360

                is_horizontal = (
                    (abs(angle_deg) <= angle_threshold)
                    or (abs(angle_deg - 180) <= angle_threshold)
                    or (abs(angle_deg + 180) <= angle_threshold)
                )

                if is_horizontal:
                    if x1 <= x2:
                        left_y = y1
                        right_y = y2
                    else:
                        left_y = y2
                        right_y = y1

                    if left_y < right_y:
                        horizontal_lines_up.append([x1, y1, x2, y2])
                    elif left_y > right_y:
                        horizontal_lines_down.append([x1, y1, x2, y2])
                    else:
                        horizontal_lines_up.append([x1, y1, x2, y2])

    horizontal_lines_up = np.array(horizontal_lines_up)
    horizontal_lines_down = np.array(horizontal_lines_down)

    return horizontal_lines_up, horizontal_lines_down


