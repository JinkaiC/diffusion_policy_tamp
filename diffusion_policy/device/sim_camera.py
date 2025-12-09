import rospy
from sensor_msgs.msg import Image
import numpy as np

class SimCamera:
    def __init__(
        self,
        topic='/genesis/color_image'
    ):
        super(SimCamera, self).__init__()
        rospy.Subscriber(topic, Image, self.image_callback)
        self.image_array = None
        
    def image_callback(self, msg):
        height = msg.height
        width = msg.width
        encoding = msg.encoding

        # 根据编码确定通道数
        if encoding in ['rgb8', 'bgr8']:
            channels = 3
        elif encoding == 'mono8':
            channels = 1
        else:
            rospy.logwarn(f"不支持的编码格式: {encoding}")
            return
        image_data = np.frombuffer(msg.data, dtype=np.uint8)
        self.image_array = image_data.reshape((height, width, channels))[:, :, ::-1]
    
    def get_rgb_image(self):
        return self.image_array

    def get_depth_image(self):
        return None

    def get_rgbd_image(self):
        return self.get_rgb_image(), self.get_depth_image()
