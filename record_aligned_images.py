import pyrealsense2 as rs
import numpy as np
import time
import cv2
import os

W = 640
H = 480
save_path = 'data/3m_top/cam1'

# Create folders
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print('create {}'.format(save_path))
save_path_depth = os.path.join(save_path, 'depth')
save_path_color = os.path.join(save_path, 'color')
if not os.path.exists(save_path_depth):
    os.mkdir(save_path_depth)
if not os.path.exists(save_path_color):
    os.mkdir(save_path_color)

# Get current time
def get_current_time():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d_%H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# Alignment
align_to = rs.stream.color
alignedFs = rs.align(align_to)

# Start streaming
pipeline.start(config)

idx = 0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        fs = pipeline.wait_for_frames()
        frames = alignedFs.process(fs)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images1 = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images1)
        # Save images
        t = get_current_time()
        np.save(os.path.join(save_path_depth, '{:05d}_{}.npy'.format(idx, t)), depth_image)
        cv2.imwrite(os.path.join(save_path_color, '{:05d}_{}.jpg'.format(idx, t)), color_image)

        idx += 1
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
