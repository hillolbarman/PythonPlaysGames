import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time = time.time()
while(True):
    
    printscreen_pil =  ImageGrab.grab(bbox = (0, 35, 800, 636))
    printscreen_numpy =   np.array(printscreen_pil) 
    printscreen_rgb = cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB)
    cv2.imshow('window',printscreen_rgb)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    print("Loop time: {} seconds".format(time.time()-last_time))
    last_time = time.time()