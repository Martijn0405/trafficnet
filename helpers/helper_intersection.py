import numpy as np

def compute_vanishing_point(lines, y_value=None):
    """
    Estimate the vanishing point from a set of lines in the form [x1, y1, x2, y2].
    If y_value is provided, fix the y-coordinate and solve for the best x-coordinate.
    """
    A = []
    b = []

    if len(lines) == 0:
        return None, None
    
    for line in lines:
        x1, y1, x2, y2 = line
        # Line equation: a*x + b*y + c = 0
        a = y1 - y2
        b_ = x2 - x1
        c = x1 * y2 - x2 * y1
        
        # Normalize (optional but helps numerically)
        norm = np.sqrt(a**2 + b_**2)
        if norm == 0:
            continue
        a /= norm
        b_ /= norm
        c /= norm
        
        if y_value is not None:
            # Fix y = y_value, solve for x: a*x + b_*y_value + c = 0
            # So: a*x = -b_*y_value - c
            A.append([a])
            b.append([-b_ * y_value - c])
        else:
            # Add to system (Ax = b form) for both x and y
            A.append([a, b_])
            b.append([-c])
    
    A = np.array(A)
    b = np.array(b)
    
    if y_value is not None:
        # Solve for x only: A*x = b
        vp_x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        vp_x_flat = vp_x.flatten()
        print(f"- Vanishing point: ({vp_x_flat[0]:.2f}, {y_value:.2f})")
        return vp_x_flat[0], y_value
    else:
        # Least squares solution to minimize ||Ax + b|| for both x and y
        vp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        vp_flat = vp.flatten()
        print(f"- Vanishing point: ({vp_flat[0]:.2f}, {vp_flat[1]:.2f})")
        return vp_flat[0], vp_flat[1]