import numpy as np
from helpers.helper_size import compute_scene_depth

if __name__ == "__main__":
    src_pts = np.float32([[197.78, 1052.53], [2274.87, 1052.53], [605.08, 538.04], [1068.36, 538.04]])
    depth_function = compute_scene_depth(src_pts)
    depth = depth_function([197.78, 1052.53], [405.08, 745.04])
    print(f"- Depth: {depth:.2f}")