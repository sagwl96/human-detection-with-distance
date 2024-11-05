# human-detection-with-distance
A simple python program to use intel realsense rgbd camera and yolo v8 model to detect humans and get the distance to them


Example usage
```python
from pedestriandetector import PedestrianDetector

detector = PedestrianDetector(using_realsense=True)
ped_info = detector.get_ped_info()
print(ped_info)
detector.release_resources()
```

Note: The program earlier uses the focal length for Intel Realsense D415 RGBD camera. If you have any other depth camera, you would need to find the focal length (in pixels) to use this program correctly.
