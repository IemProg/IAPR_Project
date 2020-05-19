import imageio
from skimage import color
from PIL import Image
import math
import numpy as np
from skimage import morphology
from skimage.transform import resize
import scipy
import gzip
import random

image_len = 28
image_wid = 28
n_augmentation = 50

def rescale_down_sample(img, new_len, new_wid):
    img = resize(img, (new_len, new_wid))
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

def apply_all(img, theta, unzoom, dx, dy, image_len, image_wid):
    return resize(translate(unzoom_image(rotate_image(img, theta),unzoom),dx,dy), (image_len, image_wid))

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
    multiply_image = color.rgb2gray(imageio.imread(path + "dot.png"))
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
        augmented_plus = apply_all(plus_image_ds,theta, unzoom, dx, dy, image_len, image_wid)
        augmented_minus = apply_all(minus_image_ds,theta, unzoom, dx, dy, image_len, image_wid)
        augmented_multiply = apply_all(multiply_image_ds,theta, unzoom, dx, dy, image_len, image_wid)
        augmented_divide = apply_all(divide_image_ds,theta, unzoom, dx, dy, image_len, image_wid)
        augmented_equal = apply_all(equal_image_ds,theta, unzoom, dx, dy, image_len, image_wid)
        augmented_data_set['+'].append(augmented_plus)
        augmented_data_set['-'].append(augmented_minus)
        augmented_data_set['*'].append(augmented_multiply)
        augmented_data_set['/'].append(augmented_divide)
        augmented_data_set['='].append(augmented_equal)

    return augmented_data_set


def data_labeled(data):
    """
    A function to generate dataset
    Input: data is dictionary contrains the key as labels and their values as a list of images
            Data is given by generate_data function
    output: X narray, and Y labels for each row in X
    """

    data_labeled = np.zeros((len(data['+'])*5, 28, 28))
    labels = np.zeros((len(data['+'])*5, 1))
    classes = {'+':0, '-': 1, '*':2, '/':3, '=':4}
    
    shift = 0
    for i in data.keys():
        for k in range(len(data[i])):
            data_labeled[k+shift] = data[i][k]
            labels[k+shift] = classes[i]
        #We need a shift in order to avoid overwritting samples, each new key we start at zero    
        shift += 100
    return data_labeled, labels

def concatenate_dataset(mnist, operators):
    """
    inputs: - mnist[0]: train_data,  mnist[1]:train_labels , mnist[2]: test_data,  mnist[3]:test_labels 
            - operators[0]: train_data,  operators[1]:train_labels, operators[2]: train_data,  operators[3]:train_labels  <<<=== data_oper, labels_oper = data_labeled(data_operators)
    output: train_imgs, train_labels, test_imgs, test_labels
    """
    train_imgs = np.concatenate((operators[0], mnist[0]), axis=0)
    train_labels = np.concatenate((operators[1], mnist[1]), axis=0)

    test_imgs = np.concatenate((operators[2], mnist[1]), axis=0)
    test_labels =  np.concatenate((operators[3], mnist[3]), axis=0)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    return  train_imgs, train_labels, test_imgs, test_labels

def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def load_mnist(data_folder = "./mnist"):
    image_shape = (28, 28)
    train_set_size = 60000
    test_set_size = 10000

    train_images_path = os.path.join(data_folder, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_folder, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_folder, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz')

    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)
    #mnist[0]: train_data,  mnist[1]:train_labels , mnist[2]: test_data,  mnist[3]:test_labels 
    mnist = [train_images, train_labels, test_images, test_labels]
    return mnist