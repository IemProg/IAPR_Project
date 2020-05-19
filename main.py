import cv2
import os, sys, time

import argparse

#3rd party library
from utils import *


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="Output path to save video")

args = vars(ap.parse_args())

#print(args)

video_path = args['input']
saving_path = args['output']

if video_path[-3:] == "avi" or video_path[-3:] == "mp4":
    print("We are in: ", video_path)
    frames = Video_to_Frames(video_path)
    print("Done extracting frames from {}, number of frames is: {}".forma(video_path, len(frames)) )
    print("-----------------------------------------------------------------------")