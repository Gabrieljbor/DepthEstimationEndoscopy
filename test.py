import os
import glob
import argparse
import matplotlib
import numpy as np

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images, load_inputs
from matplotlib import pyplot as plt

from PIL import Image
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import imageio
import cv2


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='model5.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
##inputs = load_inputs("data/seq_3/rgb.txt")
##inputs = load_inputs("numbers.txt")
##inputs = load_inputs("color.txt")
inputs = load_images( glob.glob(args.input) )


print("inputs")
#print(inputs)
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
print("outputs")
#print(outputs)


# Get the gts
##gts = []
##with open("data/seq_3/depth.txt", "r") as file:
##    for line in file:
##        exrFileName = line.strip()
##        exrFileName = exrFileName.replace("depth", "gts")
##        exr_image = cv2.imread(exrFileName, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
##        gray = exr_image[..., 2]
##        gray *= 10
##        inverted_array = 1 - gray
##        scaled_array = (inverted_array * 255).astype(np.uint8)
##        image = Image.fromarray(scaled_array, mode='L')
##        resized_img1 = image.resize((320, 240))
##        array = np.asarray(resized_img1)/255.0
##        reshaped_array = np.expand_dims(array, axis=-1)
##        gts.append(reshaped_array)
##gts = np.array(gts)
##print("gts")
###print(gts)
##


##gts = []
##with open("depth.txt", "r") as file:
##    for line in file:
##        fileName = line.strip()
##        img1 = Image.open(fileName).convert("L")
##        resized_img1 = img1.resize((320, 240))
##        array = np.asarray(resized_img1)/255.0
##        reshaped_array = np.expand_dims(array, axis=-1)
##        gts.append(reshaped_array)
##gts = np.array(gts)
##print("gts")
##
##        



# Compute errors
##abs_rel = []
##rmse = []
##for i in range(len(gts)):
##    gt = gts[i]
##    pred = outputs[i]
##            
##    abs_rel_i = np.mean(np.abs(gt - pred) / gt)
##    abs_rel.append(abs_rel_i)
##            
##    rmse_i = (gt - pred) ** 2
##    rmse_i = np.sqrt(rmse_i.mean())
##    rmse.append(rmse_i)
##
##print("abs_rel")
##print(abs_rel)
##print("rmse")
##print(rmse)
##print(sum(rmse)/len(rmse))


#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
display_images(outputs.copy(), inputs.copy())

##
##def to_multichannel(i):
##    if i.shape[2] == 3: return i
##    i = i[:,:,0]
##    return np.stack((i,i,i), axis=2)
##
##
##import skimage
##from skimage.transform import resize
##
##
##shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
##
##all_images = []
##
##for i in range(outputs.shape[0]):
##    imgs = []
##
##    rescaled = outputs[i][:,:,0]
##
##    rescaled = rescaled - np.min(rescaled)
##    rescaled = rescaled / np.max(rescaled)
##    imgs.append(rescaled)
##
##
##    img_set = np.hstack(imgs)
##    all_images.append(img_set)
##
##all_images = np.stack(all_images)
##for i in range(len(all_images)):
##    im = Image.fromarray(np.uint8(all_images[i] * 255.0))
##    im.save("output/file%s.jpeg" % i)
