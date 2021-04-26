'''
Authour:    Padraig McGloin
Date:       26/04/2021
Project:    Trajectory Prediction System
Filename:   pyKinectProjectilePrediction.py
'''
# package imports
import threading
import numpy
import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

''' Kalman Filter Class '''
class KalmanFilter(threading.Thread):

    """ Class initalisation """
    def __init__(self, name=None):
        #initialise thread
        threading.Thread.__init__(self)
        self.name = name
        # initialise OpenCV Kalman filter
        self.kalmanFilter = cv2.KalmanFilter(4, 2)
        # add measurement matrix to Kalman filter
        self.kalmanFilter.measurementMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0]], numpy.float32)
        # add transition matrix to Kalman filter
        self.kalmanFilter.transitionMatrix = numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)

    """ Estimates X & Y coordinates via 2D Kalman filter """
    def estimate(self, measuredX, measuredY):
        # creates numpy array of measured values.
        measured = numpy.array([[numpy.float32(measuredX)], [numpy.float32(measuredY)]])
        #Updates the predicted state from the measurement.
        self.kalmanFilter.correct(measured)
        #Computes a predicted state.
        predicted = self.kalmanFilter.predict()
        return predicted

''' Object Detection Class '''
class ObjectDetection(threading.Thread):

    """ Class initalisation """
    def __init__(self, kalmanFilterObj, name=None):
        #initialise thread
        threading.Thread.__init__(self)
        self.name = name
        self.kalmanFilterObj = kalmanFilterObj
        # initialise OpenCV background subtractor with a history of 100 frames and varience threshold of 40
        self.openCVBackgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        self.firstFrame = None

    """ Performs various image processing techniques to extract X, Y, W & H coordinates, feeds data to Kalman filter and displays results """
    def objectDetection(self, inputFrame_8bit, inputFrame_16bit):
        # required for Open CV imshow()
        done = False
        while not done:
            # apply Open CV MOG2 background subtractor
            mask = self.openCVBackgroundSubtractor.apply(inputFrame_8bit)
            # apply Gaussian blur
            blurImage = cv2.GaussianBlur(mask, (23, 23), 0)
            # if first frame is None
            if self.firstFrame is None:
                # set first frame
                self.firstFrame = blurImage
                # get background distance from center pixel of 16bit frame, height then width coordinated
                self.backgroundDistance = inputFrame_16bit[212,256]
                self.measuredDistance = 0

            # create frame delta frame
            frameDelta = cv2.absdiff(self.firstFrame, blurImage)
            # create binary thresold image values greater than 20 = 255
            thresholdImage = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
            # dilate threshold image
            dilateImage = cv2.dilate(thresholdImage, None, iterations=1)
            # find contours from threshold image
            contours = cv2.findContours(thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # extract contours
            contours = contours[0]
            index = 0
            filteredContours = []
            for contour in contours:
                #limit bounding box area to values between 40 and 200000
                if cv2.contourArea(contour) < 40 or cv2.contourArea(contour) > 20000:
                    del contours[index]
                else:
                    filteredContours.append(contours[index])
                index = index + 1

            # multiple object detection disabled
            if len(filteredContours) != 1:
                done = True

            # colourspace change to add colour to greyscale image, BGR
            inputFrame_8bit = numpy.dstack((inputFrame_8bit,inputFrame_8bit, inputFrame_8bit))
            for contour in filteredContours:
                # extract measured X & Y coordinates and W & H of bounding area
                (measuredX, measuredY, measuredW, measuredH) = cv2.boundingRect(contour)
                try:
                    #extract distance from origional 16-bit data, y then x coordinated
                    self.measuredDistance = inputFrame_16bit[measuredY + int(measuredW/2), measuredX + int(measuredH/2)]
                except Exception:
                    continue

                # send measured X & Y coordinates for prediction
                predictedCoords = self.kalmanFilterObj.estimate(measuredX, measuredY)
                predictedX = int(predictedCoords[0])
                predictedY = int(predictedCoords[1])
                # create vector for neural network
                # draw on 16-bit and 8-bit frames
                cv2.circle(inputFrame_8bit, (measuredX + int(measuredW/2), measuredY + int(measuredH/2)), 2, [0,255,0], 2)
                cv2.circle(inputFrame_16bit, (measuredX + int(measuredW/2), measuredY + int(measuredH/2)), 2, 65535, 2)
                cv2.rectangle(inputFrame_8bit, (measuredX, measuredY), (measuredX + measuredW, measuredY + measuredH), (255, 0, 0), 3)
                cv2.line(inputFrame_8bit,(measuredX, measuredY) , (measuredX - 50, measuredY - 20), [255,0,0], 2,8)
                cv2.putText(inputFrame_8bit, "Actual", (measuredX - 50, measuredY - 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [255,0,0])
                cv2.rectangle(inputFrame_8bit, (predictedX, predictedY), (predictedX + measuredW, predictedY + measuredH), (0, 255, 255), 3)
                cv2.line(inputFrame_8bit,(predictedX + measuredW, predictedY + measuredH) , (predictedX + measuredW + 50, predictedY + measuredH + 20), [0,255,255], 2,8)
                cv2.putText(inputFrame_8bit, "Predicted", (predictedX + measuredW + 50, predictedY + measuredH + 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,255])

            cv2.putText(inputFrame_8bit, ("Background Distance: " + str(self.backgroundDistance) + "mm"), (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,0])
            cv2.putText(inputFrame_8bit, ("Distance: " + str(self.measuredDistance) + "mm"), (20,400), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,0])
            # Display image feeds
            # inputFrame -> mask -> delta -> threshold -> dilate
            cv2.imshow("8-Bit Image Feed", inputFrame_8bit)
            cv2.moveWindow("8-Bit Image Feed", 0, 0)
            cv2.imshow("Mask Feed", mask)
            cv2.moveWindow("Mask Feed", 520, 0)
            cv2.imshow("Frame Delta Feed", frameDelta)
            cv2.moveWindow("Frame Delta Feed", 1040, 0)
            cv2.imshow("Threshold Feed", thresholdImage)
            cv2.moveWindow("Threshold Feed", 0, 460)
            cv2.imshow("Dilate Feed", dilateImage)
            cv2.moveWindow("Dilate Feed", 520, 460)
            cv2.imshow("16-Bit Image Feed", inputFrame_16bit)
            cv2.moveWindow("16-Bit Image Feed", 1040, 460)
            # if the `q` key is pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            done = True

''' Kinect Frame Scaling Class '''
class ScaleFrame(threading.Thread):
    
    """ Class initalisation """
    def __init__(self, objectDetectionObj, name=None):
        #initialise thread
        threading.Thread.__init__(self)
        self.name = name
        self.objectDetectionObj = objectDetectionObj
        # initialise Kinect V2 to retrieve depth frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        # retrive Kinect frame width & height
        self._kinect_Frame_Width = self._kinect.depth_frame_desc.width
        self._kinect_Frame_Height = self._kinect.depth_frame_desc.height

    """ Reshapes Kinect 16-bit linear array into 16-bit & 8-bit 2D arrays """
    def scaleFrame(self):
        # reshape array
        reshapedArray = numpy.reshape(self._kinect.get_last_depth_frame(), (self._kinect_Frame_Height, self._kinect_Frame_Width))
        # normalise values in array between 0 and 255
        scaledImage = cv2.normalize(reshapedArray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # conversion to absolute 8-bit values
        convertedFrame = cv2.convertScaleAbs(scaledImage)
        # call object detection passing 8-bit and 16-bit 2D arrays
        self.objectDetectionObj.objectDetection(convertedFrame, reshapedArray)

''' Main Class '''
class PyKinectProjectilePredictionRuntime(threading.Thread):

    """ Class initalisation """
    # def __init__(self, scaleFrameObj, name=None):
    def __init__(self, name=None):
        threading.Thread.__init__(self)
        self.name = name
        self._done = False
        #initialise Kalman Filter thread
        self.kalmanFilterObj = KalmanFilter(name='KalmanFilterThread')
        self.kalmanFilterObj.daemon = True
        self.kalmanFilterObj.start()
        self.kalmanFilterObj.join()
        # initialise object detection thread and pass kalman filter thread object
        self.objectDetectionObj = ObjectDetection(self.kalmanFilterObj, name='ObjectDetectionThread')
        self.objectDetectionObj.daemon = True
        self.objectDetectionObj.start()
        self.objectDetectionObj.join()
        # initialise scale frame thread and pass object detection thread object
        self.scaleFrameObj = ScaleFrame(self.objectDetectionObj, name='ScaleFrameThread')
        self.scaleFrameObj.daemon = True
        self.scaleFrameObj.start()
        self.scaleFrameObj.join()

    """ Calls scale frame on new depth frame arrival """
    def run(self):
        while not self._done:
            if self.scaleFrameObj._kinect.has_new_depth_frame():
                self.scaleFrameObj.scaleFrame()

''' Initalisation Code '''
if __name__ == '__main__':
    # initialise pyKinectProjectilePrediction main
    mainObj = PyKinectProjectilePredictionRuntime(name='Main')
    mainObj.daemon = True
    mainObj.start()
    mainObj.join()
