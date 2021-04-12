# FFMPEG ... FFV1 codec is lossless and supports gray16le pixel format
import threading
import cv2
import numpy

class DrawFrameOpenCVRuntime(threading.Thread):
    def __init__(self, inputFrameQueue, _kinect, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        # super(DrawFrameRuntime,self).__init__()
        threading.Thread.__init__(self)
        self.inputFrameQueueDraw = inputFrameQueue
        self._kinect = _kinect
        self.target = target
        self.name = name
        self._kinect_Frame_Width = self._kinect.depth_frame_desc.width
        self._kinect_Frame_Height = self._kinect.depth_frame_desc.height
        # Loop until the user clicks the close button.
        self._done = False
    
    def run(self):
        while not self._done:
            if not self.inputFrameQueueDraw.empty():
                inputFrame = self.inputFrameQueueDraw.get()
                # print("1\n")
                # print(inputFrame)
                self.reshapedArray = numpy.reshape(inputFrame, (self._kinect_Frame_Height, self._kinect_Frame_Width))
                # print("2\n")
                # print(self.reshapedArray)
                self.img_scaled = cv2.normalize(self.reshapedArray, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
                # print("3\n")
                # print(self.img_scaled)
                cv2.imshow("frame", self.img_scaled)  
                if cv2.waitKey(1) == ord('q'):
                    break
                cv2.destroyAllWindows()