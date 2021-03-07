import numpy as np
from PIL import ImageGrab
import cv2

def grabScreen():    
    printscreen_pil =  ImageGrab.grab(bbox = (0, 35, 400, 335))
    printscreen_numpy =   np.array(printscreen_pil) 
    printscreen_rgb = cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB)
    return printscreen_rgb