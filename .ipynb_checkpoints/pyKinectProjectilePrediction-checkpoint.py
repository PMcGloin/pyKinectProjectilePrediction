from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import ctypes
import _ctypes
import pygame
import sys
import numpy as np
import cv2

#if sys.hexversion >= 0x03000000:
#    import _thread as thread
#else:
#    import thread
class DepthRuntime(object):
    def __init__(self):
        pygame.init()
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
        # Loop until the user clicks the close button.
        self._done = False
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
        # back buffer surface for getting Kinect depth frames, 8bit grey, width and height equal to the Kinect depth frame size
        self._frame_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        pygame.display.set_caption("Kinect for Windows v2 Depth")
    #def background_subtraction(self, current_frame, previous_frame):
    #    previousFrame = [0] * 217088
    #    return frame
    def draw_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        frame8bit=np.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()
    def run(self):
        # -------- Main Program Loop -----------
        
        frame = [0] * 217088
        frames = [frame] * 5
        fgbg = cv2.createBackgroundSubtractorKNN()
        # fgbg = cv2.createBackgroundSubtractorMOG2()
        # print (len(previousFrames))
        # print(previousFrames)
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop
                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
            # --- Getting frames and drawing 
            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                fgmask = fgbg.apply(frame)
                # flattenMask = []
                # for item in fgmask:
                #     flattenMask.append(item)
                flattenMask = [value for element in fgmask for value in element]
                # print (type(flattenMask[0]))
                flattenMask = np.array(flattenMask)
                # flattenMask = np.array(fgmask)
                # flattenMask = flattenMask / 255
                # print ("flattenMask\n",flattenMask)
                frameMask = []
                # frameMask = np.array(frameMask)
                for val in np.nditer(flattenMask):
                    # i = 0
                    if val == 255:
                        frameMask.append(1)
                        # val = 1
                    else:
                        frameMask.append(0)
                        # val = 0
                    # i += 1
                frameMask = np.array(frameMask)
                # np.set_printoptions(threshold=sys.maxsize)
                # print("frame\n",frame)
                # print ("flattenMask\n",flattenMask)
                # print ("frameMask\n",frameMask)
                outputFrame = np.multiply(frame, frameMask)
                # frames.append(outputFrame)
                # frames.pop(0)
                # outputFrame2 = []
                # cv2.fastNlMeansDenoisingMulti(frames, 4, 4, outputFrame2)
                # outputFrame2 = cv2.fastNlMeansDenoising(outputFrame)
                # outputFrame = np.multiply(frame, fgmask)
                # cv2.imshow('frame',fgmask)
                self.draw_depth_frame(outputFrame, self._frame_surface)
                # k = cv2.waitKey(30) & 0xff
                # if k == 27:
                #     break
                # frames.append(frame)
                # frames.pop(0)
                # outputFrame = np.subtract(frames[0], frames[1])
                # self.draw_depth_frame(outputFrame, self._frame_surface)
                #self.draw_depth_frame(frame, self._frame_surface)
                #frame = np.average(np.array([frame, previousFrame]), axis=0)
                #np.set_printoptions(threshold=sys.maxsize)
                #print(outputFrame)
                #print(frame.size)
                # outputFrame = (np.array(previousFrames[0]) + np.array(previousFrames[1]) + np.array(previousFrames[2]) + np.array(previousFrames[3]) + np.array(previousFrames[4])) / 5
                # self.draw_depth_frame(outputFrame.astype(int), self._frame_surface)
                # frame2 = cv.fastNlMeansDenoisingMulti(previousFrames, 2 , 3)

                frame = None
                outputFrame = None

            self._screen.blit(self._frame_surface, (0,0))
            pygame.display.update()
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
            # --- Limit to 60 frames per second
            self._clock.tick(60)
        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()
__main__ = "Kinect v2 Depth"
game =DepthRuntime();
game.run();