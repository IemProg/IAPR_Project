import imageio
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage import morphology
import scipy
import random

image_len = 28
image_wid = 28
n_augmentation = 50


def rescale_down_sample(img, new_len, new_wid):
    length, width = img.shape
    factor_len, remainder_len = divmod(length, new_len)
    factor_wid, remainder_wid = divmod(width, new_wid)
    off_set_len_inf = math.floor((length-(new_len*factor_len))/2)
    off_set_len_sup = math.ceil((length-(new_len*factor_len))/2)
    off_set_wid_inf = math.floor((width-(new_wid*factor_wid))/2)
    off_set_wid_sup = math.ceil((width-(new_wid*factor_wid))/2)
    img = img[off_set_len_inf:length-off_set_len_sup:factor_len,off_set_wid_inf:width - off_set_wid_sup:factor_wid]
    assert img.shape == (new_len,new_wid), "Image is " + str(img.shape)
    return img

def rotate_image(img, theta):
    pil_img = Image.fromarray(np.uint8((1-img)*255), "L")
    rotated_img = pil_img.rotate(theta)
    return (255 - np.array(rotated_img))/255.

def unzoom_image(img, unzoom):
    assert unzoom <= 1,"unzoom must be <= 1"
    length, width = img.shape
    pad_len_inf = math.floor((28/unzoom-length)/2)
    pad_len_sup = math.ceil((28/unzoom-length)/2)
    pad_wid_inf = math.floor((28/unzoom-width)/2)
    pad_wid_sup = math.ceil((28/unzoom-width)/2)
    padded_img = np.pad(img, ((pad_len_inf, pad_len_sup),(pad_wid_inf, pad_wid_sup)), constant_values = 1)
    return scipy.ndimage.zoom(padded_img, unzoom, cval=1)

def translate(img, dx, dy):
    length, width = img.shape
    while min(list(set(img[:, width - abs(dx):].flatten()))) < 0.98:
        dx -=1
        if dx == 1:
            break
    while min(list(set(img[length-abs(dy):, :].flatten()))) <0.98:
        dy -=1
        if dy == 1:
            break
    x_moved = np.roll(img, dx, 1)
    return np.roll(x_moved, dy, 0)

def apply_all(img, theta, unzoom, dx, dy):
    return translate(unzoom_image(rotate_image(img, theta),unzoom),dx,dy)

path = "operators/"


def generate_data(path = "operators/", image_len = 28, image_wid = 28, n_augmentation = 50):
    """
    path : Root path to images folder
    image_len, image_wid:  parameters of the out images
    n_augmentation: number os generated samples for each images

    output: a dictionary with contains as key the signs, each key has a list of narrays.
    """
    plus_image = color.rgb2gray(imageio.imread(path + "+.png"))
    minus_image = color.rgb2gray(imageio.imread(path + "-.png"))
    multiply_image = color.rgb2gray(imageio.imread(path + "*.png"))
    divide_image = color.rgb2gray(imageio.imread(path + "%.png"))
    equal_image = color.rgb2gray(imageio.imread(path + "=.png"))

    plus_image_ds = rescale_down_sample(plus_image, image_len, image_wid)
    minus_image_ds = rescale_down_sample(minus_image, image_len, image_wid)
    multiply_image_ds = rescale_down_sample(multiply_image, image_len, image_wid)
    divide_image_ds = rescale_down_sample(divide_image, image_len, image_wid)
    equal_image_ds = rescale_down_sample(equal_image, image_len, image_wid)


    augmented_data_set = {'+':[],'-':[],'*':[],'/':[],'=':[]}
    for i in range(n_augmentation):
        theta = random.randint(0,360) if random.uniform(0,1) > 0.2 else 0
        unzoom = random.uniform(0.5, 1) if random.uniform(0,1) > 0.2 else 1
        dx = random.randint(1, 5) if random.uniform(0,1) > 0.2 else 1
        dy = random.randint(1, 5) if random.uniform(0,1) > 0.2 else 1
        augmented_plus = apply_all(plus_image_ds,theta, unzoom, dx, dy)
        augmented_minus = apply_all(minus_image_ds,theta, unzoom, dx, dy)
        augmented_multiply = apply_all(multiply_image_ds,theta, unzoom, dx, dy)
        augmented_divide = apply_all(divide_image_ds,theta, unzoom, dx, dy)
        augmented_equal = apply_all(equal_image_ds,theta, unzoom, dx, dy)
        augmented_data_set['+'].append(augmented_plus)
        augmented_data_set['-'].append(augmented_minus)
        augmented_data_set['*'].append(augmented_multiply)
        augmented_data_set['/'].append(augmented_divide)
        augmented_data_set['='].append(augmented_equal)

    return augmented_data_set