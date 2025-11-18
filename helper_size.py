from math import sqrt

def compute_trapezoid_width_function(point1, point2, point3, point4):
    """
    Creates a linear function for the width of the trapezoid formed by 4 points.
    The trapezoid has:
    - Bottom edge: point1 to point2 (at y = point1_y)
    - Top edge: point3 to point4 (at y = point3_y)
    
    Args:
        point1: Tuple (x, y) - leftmost bottom point
        point2: Tuple (x, y) - rightmost bottom point
        point3: Tuple (x, y) - leftmost top point
        point4: Tuple (x, y) - rightmost top point
    
    Returns:
        Function width(y) that returns the width of the trapezoid at a given y coordinate
    """
    point1_x, point1_y = point1
    point2_x, _ = point2
    point3_x, point3_y = point3
    point4_x, _ = point4
    
    # Calculate width at bottom (y = point1_y)
    width_bottom = point2_x - point1_x
    
    # Calculate width at top (y = point3_y)
    width_top = point4_x - point3_x
    
    # Create linear function: width(y) = a * y + b
    # We have two points: (point1_y, width_bottom) and (point3_y, width_top)
    if abs(point3_y - point1_y) < 1e-6:
        # If y values are the same, return constant function
        a = 0.0
        b = width_bottom
        def width(y):
            return width_bottom
    else:
        # Linear interpolation: a = (width_top - width_bottom) / (point3_y - point1_y)
        # b = width_bottom - a * point1_y
        a = (width_top - width_bottom) / (point3_y - point1_y)
        b = width_bottom - a * point1_y
        
        def width(y):
            return a * y + b
    
    print(f"- Linear function: width(y) = {a:.6f} * y + {b:.6f}")
    
    return width

def compute_lines_width(lines_bottom_front):
    """
    Computes the width for each line in lines_bottom_front.
    The width is the absolute distance between x1 and x2 for the 2 points on each line.
    
    Args:
        lines_bottom_front: List of lines, where each line is a list of 2 points [[x1, y1], [x2, y2]]
    
    Returns:
        List of widths (one for each line), where width = abs(x2 - x1)
    """
    lines_bottom_width = []
    
    for line in lines_bottom_front:
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
        lines_bottom_width.append([point1[1], width])
    
    return lines_bottom_width

def compare_line_widths_to_function(lines_bottom_width, width_function):
    """
    Compares the actual width of each line to the expected width from the trapezoid width function.
    
    Args:
        lines_bottom_width: List of [y, width] pairs, where y is the y-coordinate and width is the actual width
        width_function: Function that takes a y-coordinate and returns the expected width
    
    Returns:
        List of comparisons, where each element is [y, actual_width, expected_width, difference, ratio]
    """
    comparisons = []
    
    for line in lines_bottom_width:
        if line is None or len(line) != 2:
            continue
        
        y = line[0]
        actual_width = line[1]
        
        # Get expected width from the function
        expected_width = width_function(y)
        
        # Calculate difference and ratio
        difference = actual_width - expected_width
        ratio = actual_width / expected_width if expected_width != 0 else float('inf')
        
        comparisons.append([y, actual_width, expected_width, difference, ratio])

    # Compute average ratio
    average_ratio = None
    ratios = [comp[4] for comp in comparisons if comp[4] != float('inf')]
    if len(ratios) > 0:
        average_ratio = sum(ratios) / len(ratios)
        print(f"[Size] Average ratio: {average_ratio:.4f} (from {len(ratios)} valid comparisons)")
    
    return average_ratio

def compute_scene_depth(percentage, point1, point2, vp_center_x, vp_center_y):
    """
    Computes the depth of the scene at a given percentage along the bottom line of the trapezoid.
    The depth is the length of the line from the center vanishing point to a point on the bottom line.
    
    Args:
        percentage: Float between 0 and 1, representing position along bottom line
                   0 = left edge (point1), 1 = right edge (point2)
        point1: Tuple (x, y) - leftmost bottom point of trapezoid
        point2: Tuple (x, y) - rightmost bottom point of trapezoid
        vp_center_x: X coordinate of center vanishing point
        vp_center_y: Y coordinate of center vanishing point
    
    Returns:
        Float: Distance from the point on bottom line to the center vanishing point
    """
    # Clamp percentage to [0, 1]
    percentage = max(0.0, min(1.0, percentage))
    
    # Extract coordinates
    point1_x, point1_y = point1
    point2_x, _ = point2
    
    # Calculate point on bottom line based on percentage
    # At percentage 0: point1, at percentage 1: point2
    bottom_x = point1_x + percentage * (point2_x - point1_x)
    bottom_y = point1_y  # Both points on bottom line have same y
    
    # Calculate distance from point on bottom line to center vanishing point
    dx = bottom_x - vp_center_x
    dy = bottom_y - vp_center_y
    depth = sqrt(dx**2 + dy**2)
    
    return depth

