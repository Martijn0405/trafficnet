import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_bev_depth():
    # 1. SETUP: Define Real World constraints
    # ---------------------------------------------------------
    ROAD_WIDTH_METERS = 3.5  # Standard lane width
    
    # Load your image
    # img = cv2.imread('road.jpg')
    # For this example, we create a dummy black image
    img = np.zeros((720, 1280, 3), dtype=np.uint8) 

    # 2. INPUT: The Trapezoid Coordinates (Source Points)
    # ---------------------------------------------------------
    # You found these from your vanishing point/road lines logic.
    # Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    # (These are dummy example coordinates)
    h, w = img.shape[:2]
    src_pts = np.float32([
        [580, 460],   # Top-Left (Far)
        [700, 460],   # Top-Right (Far)
        [1080, 720],  # Bottom-Right (Near)
        [200, 720]    # Bottom-Left (Near)
    ])

    # Visualize Source Trapezoid
    cv2.polylines(img, [np.int32(src_pts)], True, (0, 255, 0), 3)

    # 3. CONFIG: Define Bird's Eye View (Destination) Properties
    # ---------------------------------------------------------
    # We define a scale: How many pixels represent 1 meter?
    # Higher = more resolution, larger output image.
    PIXELS_PER_METER = 50 
    
    # Calculate the pixel width of the road in the BEV
    bev_road_width_px = int(ROAD_WIDTH_METERS * PIXELS_PER_METER)
    
    # Define the size of the output BEV image
    # We make it wide enough to hold the road, and tall enough for the depth
    bev_width = 600
    bev_height = 1000 # Arbitrary "look ahead" canvas size

    # Center the road in the destination image
    pad_x = (bev_width - bev_road_width_px) // 2

    # 4. MAPPING: Define Destination Points
    # ---------------------------------------------------------
    # We map the source trapezoid to a straight rectangle.
    # Crucial: We assume the Source Top and Source Bottom are the limits of our ROI.
    
    # Note: We map the Bottom (Near) points to the bottom of the BEV image
    # and the Top (Far) points to the top of the BEV image (0).
    # This effectively stretches the trapezoid to the full height of our canvas.
    
    dst_pts = np.float32([
        [pad_x, 0],                      # Top-Left (Far) -> y=0
        [pad_x + bev_road_width_px, 0],  # Top-Right (Far) -> y=0
        [pad_x + bev_road_width_px, bev_height], # Bottom-Right (Near) -> y=max
        [pad_x, bev_height]              # Bottom-Left (Near) -> y=max
    ])

    # 5. COMPUTE: Homography & Transformation
    # ---------------------------------------------------------
    # Calculate the Perspective Transform Matrix M
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the image to BEV
    bev_img = cv2.warpPerspective(img, M, (bev_width, bev_height))

    # 6. ESTIMATE DEPTH
    # ---------------------------------------------------------
    # Since we mapped the source trapezoid to the full 'bev_height',
    # the depth in pixels is simply the height of the image (bev_height).
    
    depth_in_pixels = bev_height
    
    # Convert to meters using our defined scale
    estimated_depth_meters = depth_in_pixels / PIXELS_PER_METER

    print("--- Results ---")
    print(f"Real Road Width: {ROAD_WIDTH_METERS}m")
    print(f"BEV Scale: {PIXELS_PER_METER} pixels/meter")
    print(f"Measured Depth (Pixels): {depth_in_pixels}px")
    print(f"Estimated Depth (Meters): {estimated_depth_meters:.2f}m")
    
    # Draw grid lines on BEV for visual verification (every 5 meters)
    for i in range(0, int(estimated_depth_meters) + 1, 5):
        y_pos = bev_height - (i * PIXELS_PER_METER)
        cv2.line(bev_img, (0, int(y_pos)), (bev_width, int(y_pos)), (0, 0, 255), 2)
        cv2.putText(bev_img, f"{i}m", (10, int(y_pos)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Show results (using plt for compatibility)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Camera View")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Bird's Eye View (Depth Estimate)")
    plt.imshow(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    estimate_bev_depth()