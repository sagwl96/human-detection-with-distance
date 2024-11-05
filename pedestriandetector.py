import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from collections import defaultdict

class PedestrianDetector:
    def __init__(self, using_realsense=False):
        self.using_realsense = using_realsense
        self.model = YOLO("yolov8m.pt")
        self.names = self.model.model.names
        self.track_history = defaultdict(lambda: [])
        self.frame_center_x = 640 / 2  # Adjust based on camera resolution
        self.focal_length = 630.1  # in pixels
        
        if using_realsense:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Error: Could not open camera.")

    def get_ped_info(self):
        pedestrian_info = []

        if self.using_realsense:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                return pedestrian_info  # Return empty if no frames are captured
            depth_image = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(color_frame.get_data())
        else:
            success, frame = self.cap.read()
            if not success:
                return pedestrian_info

        results = self.model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except:
            return pedestrian_info

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if cls == 0:  # Detect only people
                x, y, w, h = box
                depth_value = None
                angle_radians = None
                
                if self.using_realsense:
                    depth_value = depth_frame.get_distance(int(x), int(y))
                    pixel_offset = x - self.frame_center_x
                    hypotenuse = depth_value
                    opposite = (pixel_offset * hypotenuse) / np.sqrt((self.focal_length ** 2) + (pixel_offset ** 2))
                    sin_theta = opposite / hypotenuse
                    angle_radians = float(np.arcsin(sin_theta))
                
                pedestrian_info.append({
                    "id": track_id,
                    "depth": depth_value,
                    "angle": angle_radians
                })
        
        return pedestrian_info

    def release_resources(self):
        if self.using_realsense:
            self.pipeline.stop()
        else:
            self.cap.release()
        cv2.destroyAllWindows()

# Example of how to use in another Python program
# from pedestriandetector import PedestrianDetector

# detector = PedestrianDetector(using_realsense=True)
# ped_info = detector.get_ped_info()
# print(ped_info)
# detector.release_resources()
