from math import sqrt

def compute_trapezoid_width_function(src_pts):
    """
    Creates a linear function for the width of the trapezoid formed by 4 points.
    The trapezoid has:
    - Bottom edge: point1 to point2 (at y = point1_y)
    - Top edge: point3 to point4 (at y = point3_y)
    
    Args: src_pts: List of 4 points, where each point is a tuple (x, y)
    Returns: Function width(y) that returns the width of the trapezoid at a given y coordinate
    """

    point_bl_x, point_bl_y = src_pts[0]
    point_br_x, _ = src_pts[1]
    point_tl_x, point_tl_y = src_pts[2]
    point_tr_x, _ = src_pts[3]
    
    # Calculate width at bottom (y = point1_y)
    width_bottom = point_br_x - point_bl_x
    
    # Calculate width at top (y = point3_y)
    width_top = point_tr_x - point_tl_x
    
    # Create linear function: width(y) = a * y + b
    # We have two points: (point1_y, width_bottom) and (point3_y, width_top)
    if abs(point_tl_y - point_bl_y) < 1e-6:
        # If y values are the same, return constant function
        a = 0.0
        b = width_bottom
        def width(_=None):
            return width_bottom
    else:
        # Linear interpolation: a = (width_top - width_bottom) / (point3_y - point1_y)
        # b = width_bottom - a * point1_y
        a = (width_top - width_bottom) / (point_tl_y - point_bl_y)
        b = width_bottom - a * point_bl_y
        
        def width(y):
            return a * y + b
    
    print(f"- Linear function: width(y) = {a:.6f} * y + {b:.6f}")
    
    return width

def compute_lines_distance(lines):
    """
    Computes the width for each line in lines_bottom_front.
    The width is the absolute distance between x1 and x2 for the 2 points on each line.
    
    Args:
        lines_bottom_front: List of lines, where each line is a list of 2 points [[x1, y1], [x2, y2]]
    
    Returns:
        List of points and widths (one for each line), where width = abs(x2 - x1)
    """
    lines_distance = []
    
    for line in lines:
        if line is None or len(line) != 2:
            continue
        
        point1 = line[0]
        point2 = line[1]
        
        # Extract x coordinates
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
  
        # Calculate absolute distance between x1 and x2
        width = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lines_distance.append([point1, point2, width])
    
    return lines_distance

def compare_line_widths_to_function(lines_bottom_width, width_function):
    """
    Compares the actual width of each line to the expected width from the trapezoid width function.
    
    Args:
        lines_bottom_width: List of [point1, point2, width] pairs, where point1 and point2 are the points on the line and width is the actual width
        width_function: Function that takes a y-coordinate and returns the expected width
    
    Returns:
        List of comparisons, where each element is [ratio]
    """
    comparisons = []
    
    for line in lines_bottom_width:
        if line is None or len(line) != 3:
            print(f"- line: {line} is None or len(line) != 3")
            continue
        
        y = line[0][1]
        actual_width = line[2]
        
        # Get expected width from the function
        expected_width = width_function(y)
        
        # Calculate difference and ratio
        difference = actual_width - expected_width
        ratio = actual_width / expected_width if expected_width != 0 else float('inf')
        
        comparisons.append([difference, ratio])

    # Compute average ratio
    average_ratio = 0
    ratios = [comp[1] for comp in comparisons if comp[1] != float('inf')]
    if len(ratios) > 0:
        average_ratio = sum(ratios) / len(ratios)
        print(f"[Size] Average ratio flat: {average_ratio:.4f} (from {len(ratios)} valid comparisons)")
    
    return average_ratio

def compute_scene_depth(width_function):
    """
    Returns a function that takes in 2 points that cross the center vanishing point (point_a, point_b).
    Next it calculates the 2 points crossing the bottom and top line of the trapezoid (point_c, point_d).
    Line segment a-b lays on the line c-d.
    Then it calculates the fraction of the size of a-b compared to c-d.
    """

    def depth_function(point_a, point_b):
        # Sizes
        size_a_b = sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
        size_width = width_function(point_a[1])

        # Fraction of size of a-b compared to c-d
        fraction = size_a_b / size_width

        return fraction
    
    return depth_function


def compare_line_depths_to_function(lines_depth_distance, depth_function):
    """
    Compares the actual depth of each line to the expected depth from the depth function.
    
    Args:
        lines_depth_distance: List of [point1, point2, depth] pairs, where point1 and point2 are the points on the line and depth is the actual depth
        depth_function: Function that takes a y-coordinate and returns the expected depth
    
    Returns:
        List of comparisons, where each element is [ratio]
    """
    comparisons = []
    
    for line in lines_depth_distance:
        if line is None or len(line) != 3:
            print(f"- line: {line} is None or len(line) != 3")
            continue
        
        # Get ratio
        ratio = depth_function(line[0], line[1])
        comparisons.append([ratio])
    
    # Compute average ratio
    average_ratio = 0
    ratios = [comp[0] for comp in comparisons]
    if len(ratios) > 0:
        average_ratio = sum(ratios) / len(ratios)
        print(f"[Size] Average ratio depth: {average_ratio:.4f} (from {len(ratios)} valid comparisons)")
    
    return average_ratio
