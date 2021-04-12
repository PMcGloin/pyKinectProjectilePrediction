import threading
import numpy
import cv2
from queue import Queue

class BackgroundSubtractor(threading.Thread):
    # def __init__(self, inputQueue):
    def __init__(self, inputFrameQueue, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self)
        self.inputFrameQueueBackground = inputFrameQueue
        self.outputFrameQueueBackground = Queue(600)
        self.target = target
        self.name = name
        self.openCVBackgroundSubtractor = cv2.createBackgroundSubtractorKNN()
        # Loop until the user clicks the close button.
        self._done = False

    def run(self): # send frame (kinect depth [16bit] * 217088 values) and background subtractor(open cv KNN)
        while not self._done:
            if not self.inputFrameQueueBackground.empty():
                frame = self.inputFrameQueueBackground.get()
                fgmask = self.openCVBackgroundSubtractor.apply(frame)               # apply subtractor to frame
                flattenMask = [value for element in fgmask for value in element]    # flatten mask
                flattenMask = numpy.array(flattenMask)             # transform into numpy array
                frameMask = []
                for value in numpy.nditer(flattenMask):              #create mask of 1s and 0s
                    if value == 255:
                        frameMask.append(1)
                    else:
                        frameMask.append(0)
                frameMask = numpy.array(frameMask)                 # tranform into numpy array
                self.backgroundSubtractedFrame = numpy.multiply(frame, frameMask)
                self.outputFrameQueueBackground.put(self.backgroundSubtractedFrame)