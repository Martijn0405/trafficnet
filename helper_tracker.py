import numpy as np

class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox, class_id, confidence):
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'class_id': class_id,
            'confidence': confidence,
            'first_seen': True
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        if len(detections) == 0:
            for object_id in self.disappeared.keys():
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        input_centroids = []
        input_bboxes = []
        input_classes = []
        input_confidences = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
            input_bboxes.append((x1, y1, x2, y2))
            input_classes.append(int(class_id))
            input_confidences.append(conf)
            
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], 
                            input_classes[i], input_confidences[i])
        else:
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = self.objects.keys()
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                              np.array(input_centroids), axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = input_bboxes[col]
                self.objects[object_id]['class_id'] = input_classes[col]
                self.objects[object_id]['confidence'] = input_confidences[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col],
                                input_classes[col], input_confidences[col])
                                
        return self.objects


