def is_point_in_boundary_box(x1, y1, x2, y2, boundary_boxes):
    for box in boundary_boxes:
        is_in_1 = x1 >= box[0] and x1 <= box[2] and y1 >= box[1] and y1 <= box[3]
        is_in_2 = x2 >= box[0] and x2 <= box[2] and y2 >= box[1] and y2 <= box[3]
        if is_in_1 and is_in_2:
            return True
    return False