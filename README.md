# human-detection-with-distance
A simple python program to use intel realsense rgbd camera and yolo v8 model to detect humans and get the distance to them


Example usage
'''python
from pedestriandetector import PedestrianDetector

detector = PedestrianDetector(using_realsense=True)
ped_info = detector.get_ped_info()
print(ped_info)
detector.release_resources()
'''
