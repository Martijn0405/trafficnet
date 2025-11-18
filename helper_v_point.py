import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line as draw_line
from helper_intersection import compute_vanishing_point

def find_vanishing_point(acc, vx_range, vy_range):
    i, j = np.unravel_index(np.argmax(acc), acc.shape)
    vx = vx_range[j]
    vy = vy_range[i]
    return vx, vy


def line_params(x1, y1, x2, y2):
    if x2 == x1:
        m = np.inf
        b = x1  # x = b
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    return m, b

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def diamond_accumulator(lines, width, height, output_path=None, index=0):
    if lines is None: lines = []

    # Constructing test image
    factor = 5
    h = int(height / factor)
    w = int(width / factor)
    image = np.zeros((h, w))

    for line in lines:
        x1, y1, x2, y2 = line
        x1 = int(x1 / factor)
        y1 = int(y1 / factor)
        x2 = int(x2 / factor)
        y2 = int(y2 / factor)
        image[draw_line(y1, x1, y2, x2)] = 255

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)
    base = output_path + "/vp_center/diamond_" + str(index) + "_"

    # Save image
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cm.gray, aspect=1)
    plt.title('Diamond space')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.axis('image')
    plt.savefig(base + "space.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save hough transform, angles and distances
    # angle_step = 0.5 * np.diff(theta).mean()
    # d_step = 0.5 * np.diff(d).mean()
    # bounds = [
    #     np.rad2deg(theta[0] - angle_step),
    #     np.rad2deg(theta[-1] + angle_step),
    #     d[-1] + d_step,
    #     d[0] - d_step,
    # ]

    plt.figure(figsize=(10, 10))
    plt.imshow(np.log(1 + h), cmap=cm.gray, aspect=1)
    plt.title('Hough transform')
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')
    plt.axis('image')
    plt.savefig(base + "angles.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save detected lines
    lines = []
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cm.gray, aspect=1)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi / 2)
        plt.axline((x0, y0), slope=slope)
        x_real = x0 * factor
        y_real = y0 * factor
        lines.append([0, y_real - (x_real * slope), width, y_real + (width - x_real) * slope])
    plt.title('Detected lines')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.axis('image')
    plt.savefig(base + "lines.png", dpi=300, bbox_inches='tight')
    plt.close()
       
    return lines

def find_vanishing_lines_center(centroids_per_vehicle_id):
    if not centroids_per_vehicle_id:
        return []
    
    all_lines = []
    centroid_sets = [centroids_per_vehicle_id]

    for j, centroid_set in enumerate(centroid_sets):
        for vehicle_id, centroids in centroid_set.items():
            all_lines_vehicle = []
            if len(centroids) >= 2:
                for i in range(len(centroids) - 1):
                    x1 = centroids[i]['centroid_x']
                    y1 = centroids[i]['centroid_y']
                    x2 = centroids[i + 1]['centroid_x']
                    y2 = centroids[i + 1]['centroid_y']
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if line_length > 10 and line_length < 50:
                        if j == 0: all_lines_vehicle.append([x1, y1, x2, y2])
            if j == 0: all_lines.append(all_lines_vehicle)

    return all_lines

def find_vanishing_point_center(all_lines, all_lines_hough, output_path=None, w=500, h=500):
    line_sets = [all_lines, all_lines_hough]
    names = ["vehicle", "hough"]

    if len(all_lines_hough) == 0:
        return None, None
    vp_flat = compute_vanishing_point(all_lines_hough)

    if output_path:
        for j, line_set in enumerate(line_sets):
            plt.figure(figsize=(12, 8))
            for i, line in enumerate(line_set):
                x1, y1, x2, y2 = line
                color = plt.cm.viridis(i / len(line_set))
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

            # Plot vp_flat
            plt.plot(vp_flat[0], vp_flat[1], 'ro', markersize=10, label='Vanishing Point')
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
            plt.savefig(output_path + "/vp_center/hough_lines_" + names[j] + ".png", dpi=300, bbox_inches='tight')
            plt.close()

    return vp_flat[0], vp_flat[1]