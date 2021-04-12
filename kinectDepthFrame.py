import threading
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
from queue import Queue

class KinectDepthRuntime(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self)
        self.outputFrameQueueKinect = Queue(600)
        self.target = target
        self.name = name
        # # Loop until the user clicks the close button.
        self._done = False
        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

    def run(self):
        while not self._done:
            if self._kinect.has_new_depth_frame():
                # self.kinectDepthFrame = self._kinect.get_last_depth_frame() #numpy.ndarray of numpy.uint16 of 217088 values
                self.outputFrameQueueKinect.put(self._kinect.get_last_depth_frame())

    # def stop_Kinect(self):
    #     self._kinect.close()

# if __name__ == '__main__':
#     kinectObj = KinectDepthRuntime(name='kinect')
#     kinectObj.daemon = True
#     kinectObj.start()
#     kinectObj.join()
