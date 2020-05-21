import os, sys, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg

from skimage.exposure import rescale_intensity
from skimage.filters import median
import skimage.measure
import skimage.io 
from  skimage.color import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from skimage.transform import resize
from skimage.color import rgb2gray
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

def rescale_intensity_levels(img):
    low, high = np.percentile(img, (0, 20))
    return rescale_intensity(img, in_range=(low, high))

def extract_red(im, threshold):
    """
    Outputs only the red object in the image
    """
    copy = im.copy()
    mask = copy[:,:,0] > threshold[0][0]
    for i, (low_thr, high_thr) in enumerate(threshold):
        mask &= (copy[:,:,i] > low_thr)
        mask &= (copy[:,:,i] < high_thr)
    copy[~mask] = (0,0,0)
    return copy

def detect_arrow(src):
    """
    src: source image (H, W, 3)
    returns -> center : (x, y),  bounding box : [minr, minc, maxr, maxc]
                of the red arrow on top of the robot
    """
    #Increase pixels intensity 
    brighter = rescale_intensity_levels(src)
    #Extact red arrow and convert it to gray scale then denormalize it [0, 255]
    red_exctract =  rgb2gray(extract_red(brighter, ((180, 256), (-1,190), (-1,190))))*255
    gray = red_exctract.astype(int)
    #Remove paper and salt
    #gray = median(gray, disk(5))
    gray = median(gray)
    gray[gray > 0] = 255
    label_image, b = skimage.measure.label(gray, connectivity=2, return_num=True)
    label_image_overlay = label2rgb(label_image, image=src, bg_label=0)
    centers = []
    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            cx,cy = region.centroid
            cx, cy = int(cx),int(cy)
    center = (cx, cy)
    box = [minr, minc, maxr, maxc]
    assert len(center) == 2, print("No arrow detected")
    return center, box


def plot_trajectory(frames_seen, centers_seen):
    """
    to plot the trajectory of the robot according to the running frame
    """
    fig = Figure(figsize=(5, 4), dpi=180)
    # A canvas must be manually attached to the figure (pyplot would automatically
    # do it).  This is done by instantiating the canvas with the figure as
    # argument.
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    X, Y = list(zip(*centers_seen))
    fig.tight_layout(pad=0)
    ax.axis('off')
    ax.margins(0)
    ax.margins(1)

    ax.imshow(frames_seen)
    ax.set_axis_off()
    ax.plot(Y, X, "b")
    ax.plot(Y, X, "b.")

    # Option 2: Save the figure to a string.
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    #print(width, height)
    # Option 2a: Convert to a NumPy array.
    output = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    output = output[:,:,:3]
    #new_img = resize(output, (480, 720, 3))
    resized = output[:480, :720, :]
    return resized.astype(np.uint8)

def plot_trajectory2(frames_seen, centers_seen):
    """
    to plot the trajectory of the robot according to the running frame
    """
    fig = Figure(figsize=(5, 4), dpi=180)
    # A canvas must be manually attached to the figure (pyplot would automatically
    # do it).  This is done by instantiating the canvas with the figure as
    # argument.
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111, frameon=False)
    X, Y = list(zip(*centers_seen))
    ax.axis('off')
    # To remove the huge white borders
    ax.margins(0)
    ax.margins(1)
    
    ax.imshow(frames_seen)
    ax.set_axis_off()
    ax.plot(Y, X, "b")
    ax.plot(Y, X, "b.")

    # Option 2: Save the figure to a string.
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    #print(width, height)
    # Option 2a: Convert to a NumPy array.
    output = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    output = output[:,:,:3]
    #print("Output shape: ", output.shape)
    #new_img = resize(output, (480, 720, 3))
    #return new_img.astype(np.uint8)
    start_width = int((720 - 480)/2)
    start_height = int((900 - 720)/2)

    end_width = 720 - start_width
    end_height = 900 - start_height

    resized_img = output[start_width : end_width, start_height:end_height, :]
    
    
    return resized_img, output


def vidwrite(fn, images, framerate = 2, vcodec='libx264'):
    """
    fn: filename for the output video
    imags: List of numpu arrays/ or array of shape [nbr of frame, Width, height, channels]
    framerate : frames per second
    """
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=2, s='{}x{}'.format(width, height))
            #.filter('r', fps=framerate)
            #.output(fn, pix_fmt='yuv420p', vcodec=vcodec)
            .output(fn)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()