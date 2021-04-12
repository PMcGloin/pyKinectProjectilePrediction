from kinectDepthFrame import KinectDepthRuntime
from backgroundSubtraction import BackgroundSubtractor
from drawFrame import DrawFrameRuntime
# from drawFrameOpenCV import DrawFrameOpenCVRuntime

if __name__ == '__main__':

    kinectObj = KinectDepthRuntime(name='kinect')
    backgroundObj = BackgroundSubtractor(kinectObj.outputFrameQueueKinect, name='background')
    drawObj = DrawFrameRuntime(backgroundObj.outputFrameQueueBackground, kinectObj._kinect, name='draw')
    # drawObj = DrawFrameOpenCVRuntime(backgroundObj.outputFrameQueueBackground, kinectObj._kinect, name='draw')

    kinectObj.daemon = True
    backgroundObj.daemon = True
    drawObj.daemon = True

    kinectObj.start()
    backgroundObj.start()
    drawObj.start()

    kinectObj.join()
    backgroundObj.join()
    drawObj.join()
