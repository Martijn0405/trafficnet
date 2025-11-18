import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def merge_close_clusters(points, eps_merge=1.0):
    """
    Merge clusters of points that are within eps_merge distance.
    Uses DBSCAN to find clusters and then merges close ones.
    """
    # Detect clusters
    db = DBSCAN(eps=0.5, min_samples=3).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    
    clusters = []
    for label in unique_labels:
        if label == -1:
            continue
        clusters.append(points[labels == label])
    
    if len(clusters) <= 1:
        return points  # Already one cluster

    # Compute pairwise distances between clusters
    merged = []
    used = set()
    for i in range(len(clusters)):
        if i in used:
            continue
        current = clusters[i]
        for j in range(i + 1, len(clusters)):
            if j in used:
                continue
            dist = np.min(np.linalg.norm(current[:, None, :] - clusters[j][None, :, :], axis=2))
            if dist < eps_merge:
                # Merge clusters
                current = np.vstack([current, clusters[j]])
                used.add(j)
        merged.append(current)
        used.add(i)
    
    return np.vstack(merged)


# Generate two separated "islands" of points (like a car split by a pole)
np.random.seed(0)
car_left = np.random.randn(30, 2) * 0.2 + np.array([0, 0])
car_right = np.random.randn(30, 2) * 0.2 + np.array([1.0, 0])
mask_points = np.vstack([car_left, car_right])

# Before merging
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(mask_points[:,0], mask_points[:,1], color='red')
plt.title("Before merging (2 clusters)")
plt.axis('equal')

# Merge
merged_points = merge_close_clusters(mask_points, eps_merge=1.2)

# After merging
plt.subplot(1,2,2)
plt.scatter(merged_points[:,0], merged_points[:,1], color='green')
plt.title("After merging (1 merged cluster)")
plt.axis('equal')

plt.show()