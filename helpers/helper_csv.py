import pandas as pd

def export_csv(tracking_data, output_path, fps):
    if tracking_data:
        df = pd.DataFrame(tracking_data)
        df = df.sort_values(['object_id', 'frame_number']).reset_index(drop=True)
        df['timestamp_seconds'] = df['frame_number'] / fps
        excel_path = output_path.replace('.mp4', '_tracking_data.xlsx')
        df.to_excel(excel_path, index=False, sheet_name='Object_Tracking')
        summary = df.groupby('object_id').agg({
            'class_name': 'first',
            'frame_number': ['min', 'max', 'count'],
            'confidence': 'mean'
        }).round(3)
        summary.columns = ['class', 'first_frame', 'last_frame', 'detection_count', 'avg_confidence']
    else:
        print("No objects detected in the video.")