import os, sys
import cv2
import numpy as np
import glob
from tqdm import tqdm

def Video_to_Frames(Video_file):
    """
    Video_file: path to video
    it returns a list which contains frames as narray format
    """
    frames = []
    cap = cv2.VideoCapture(Video_file)  
    
    while True:
        ret, frame = cap.read()  
        if not ret:
            break # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        
    cap.release()
    return frames



def Frames_to_Video(input_frames, out_video_name = "fusion_video.avi"):
    """
    input_frames: path to frames folder
    out_video_name: video name "fusion_video.avi"
    """
    img_array = []
    for filename in tqdm((sorted(glob.glob(path+"/*.jpg"), key = os.path.getmtime))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
    #print("Size of frames is: ", img_array.shape)
    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



def resize(src, size, output_folder):
    """
    src: path to frames folder
    size: (height, width)
    output_folder: path to save images after resizing
    """

    head, tail = os.path.split(src)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    #print('Original Dimensions : ', img.shape)

    resized = cv2.resize(img, size)
    #print('Resized Dimensions ', tail, ": ", resized.shape)
    name = output_folder + "/" + tail
    cv2.imwrite(name, resized)


