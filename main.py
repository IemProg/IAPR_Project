import cv2
import os

#3rd party library
from utils import *


video_path = "./robot_parcours_1.avi"
#Video_to_Frame(video_path)

if video_path[-3:] == "avi":
    print("We are in: ", video_path)
    Video_to_Frames(video_path)
    print("Done extracting frames from ", video_path)
    print("-----------------------------------------------------------------------")