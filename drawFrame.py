import ctypes
import pygame
import numpy
import threading

class DrawFrameRuntime(threading.Thread):
    def __init__(self, inputFrameQueue, _kinect, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self)
        self.inputFrameQueueDraw = inputFrameQueue
        self._kinect = _kinect
        self.target = target
        self.name = name
        pygame.init()
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
        # Loop until the user clicks the close button.
        self._done = False
        self._kinect_Frame_Width = self._kinect.depth_frame_desc.width
        self._kinect_Frame_Height = self._kinect.depth_frame_desc.height

    def draw_depth_frame(self, incoming_Frame, target_surface):
        if incoming_Frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though
            return
        target_surface.lock()
        f8=numpy.uint8(incoming_Frame.clip(1,4000)/16.)
        frame8bit=numpy.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):
        self._frame_surface = pygame.Surface((self._kinect_Frame_Width, self._kinect_Frame_Height), 0, 24)
        self._screen = pygame.display.set_mode((self._kinect_Frame_Width, self._kinect_Frame_Height), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        pygame.display.set_caption("Kinect for Windows v2 Depth")
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop
                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
            # --- Getting frames and drawing 
            if not self.inputFrameQueueDraw.empty():
                self.draw_depth_frame(self.inputFrameQueueDraw.get(), self._frame_surface)
            self._screen.blit(self._frame_surface, (0,0))
            pygame.display.update()
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
            # --- Limit to 60 frames per second
            self._clock.tick(60)
        # Close our Kinect sensor, close the window and quit.
        pygame.quit()
