import numpy as np
import matplotlib.pyplot as plt

def export_trajectories(centroids_per_vehicle_id, output_path, w, h):
    """
    Export vehicle trajectories as line plots showing centroid paths and compute vanishing point using Hough transform.
    """
    if not centroids_per_vehicle_id:
        print("No tracking data available for vector export.")
        return
    
    plt.figure(figsize=(w/100, h/100))
    colors = plt.cm.tab20(np.linspace(0, 1, len(centroids_per_vehicle_id)))
    
    for i, (vehicle_id, centroids) in enumerate(centroids_per_vehicle_id.items()):
        if len(centroids) < 2:
            continue
        x_coords = [point['centroid_x'] for point in centroids]
        y_coords = [point['centroid_y'] for point in centroids]
        plt.plot(x_coords, y_coords, color=colors[i % len(colors)], linewidth=2, alpha=0.7, label=f'Vehicle {vehicle_id}')
        plt.scatter(x_coords[0], y_coords[0], color=colors[i % len(colors)], s=100, marker='o', label=f'Start {vehicle_id}' if i == 0 else "")
        plt.scatter(x_coords[-1], y_coords[-1], color=colors[i % len(colors)], s=100, marker='s', label=f'End {vehicle_id}' if i == 0 else "")

    plt.xlabel('X Coordinate (pixels)', fontsize=12)
    plt.ylabel('Y Coordinate (pixels)', fontsize=12)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.title('Vehicle Trajectories - Centroid Paths', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path = output_path + "/trajectories.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()