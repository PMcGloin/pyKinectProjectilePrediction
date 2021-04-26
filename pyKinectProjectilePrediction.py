import threading                        # package imports
import numpy
import cv2
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

class KalmanFilter(threading.Thread):

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

    def estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = numpy.array([[numpy.float32(coordX)], [numpy.float32(coordY)]])
        #Updates the predicted state from the measurement.
        self.kalmanFilter.correct(measured)
        #Computes a predicted state.
        predicted = self.kalmanFilter.predict()
        return predicted

class ObjectDetection(threading.Thread):

    def __init__(self, kalmanFilterObj, name=None):
        #initialise thread
        threading.Thread.__init__(self)
        self.name = name
        self.kalmanFilterObj = kalmanFilterObj
        # initialise OpenCV background subtractor with a history of 100 frames and varience threshold of 40
        self.openCVBackgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        self.firstFrame = None

    def objectDetection(self, inputFrame_8bit, inputFrame_16bit):
        done = False
        while not done:
            # inputFrame = self.convertedFrame
            mask = self.openCVBackgroundSubtractor.apply(inputFrame_8bit)
            blurImage = cv2.GaussianBlur(mask, (23, 23), 0)
            if self.firstFrame is None:
                self.firstFrame = blurImage
                # get background distance from center pixel of 16bit frame, height then width coordinated
                self.backgroundDistance = inputFrame_16bit[212,256]
                self.measuredDistance = 0

            frameDelta = cv2.absdiff(self.firstFrame, blurImage)
            thresholdImage = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
            dilateImage = cv2.dilate(thresholdImage, None, iterations=1)
            contours = cv2.findContours(thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0]
            index = 0
            filteredContours = []
            for contour in contours:
                #limit bounding box area to values between 50 and 10000
                if cv2.contourArea(contour) < 40 or cv2.contourArea(contour) > 20000:
                    del contours[index]
                else:
                    filteredContours.append(contours[index])
                index = index + 1
            # removal of multiple object detection
            if len(filteredContours) != 1:
                done = True
            # colourspace change to add colour to greyscale image, BGR
            inputFrame_8bit = numpy.dstack((inputFrame_8bit,inputFrame_8bit, inputFrame_8bit))
            # for contour in contours:
            for contour in filteredContours:
                (measuredX, measuredY, measuredW, measuredH) = cv2.boundingRect(contour)
                try:
                    #extract distance from origional 16-bit data, y then x coordinated
                    self.measuredDistance = inputFrame_16bit[measuredY + int(measuredW/2), measuredX + int(measuredH/2)]
                except Exception:
                    continue
                predictedCoords = self.kalmanFilterObj.estimate(measuredX, measuredY)
                # predictedCoords = self.kalmanFilterEstimate(measuredX, measuredY)
                predictedX = int(predictedCoords[0])
                predictedY = int(predictedCoords[1])
                #create vector for neural network
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
            #inputFrame -> mask -> delta -> threshold -> dilate
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

class ScaleFrame(threading.Thread):

    def __init__(self, objectDetectionObj, name=None):
        #initialise thread
        threading.Thread.__init__(self)
        self.name = name
        self.objectDetectionObj = objectDetectionObj
        # initialise Kinect V2 to retrieve depth frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        # retrive frame width & height
        self._kinect_Frame_Width = self._kinect.depth_frame_desc.width
        self._kinect_Frame_Height = self._kinect.depth_frame_desc.height
        
    def scaleFrame(self):
        # reshape array
        reshapedArray = numpy.reshape(self._kinect.get_last_depth_frame(), (self._kinect_Frame_Height, self._kinect_Frame_Width))
        # normalise values in array between 0 and 255
        scaledImage = cv2.normalize(reshapedArray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # conversion to absolute 8-bit values
        convertedFrame = cv2.convertScaleAbs(scaledImage)
        # call object detection passing 8-bit and 16-bit 2D arrays
        self.objectDetectionObj.objectDetection(convertedFrame, reshapedArray)
        # self.join()

class PyKinectProjectilePredictionRuntime(threading.Thread):
    """"""
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

    def run(self):
        while not self._done:
            if self.scaleFrameObj._kinect.has_new_depth_frame():
                self.scaleFrameObj.scaleFrame()

if __name__ == '__main__':
    # initialise pyKinectProjectilePrediction main
    mainObj = PyKinectProjectilePredictionRuntime(name='Main')
    mainObj.daemon = True
    mainObj.start()
    mainObj.join()
