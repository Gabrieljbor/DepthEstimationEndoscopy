from utils import DepthNorm
from io import BytesIO
from zipfile import ZipFile
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
import os
from PIL import Image, ImageOps
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import imageio
import cv2

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_nyu_data(batch_size, nyu_data_zipfile='Archive.zip'):
    data = extract_zip(nyu_data_zipfile)


    with open('Archive/train.csv', 'r') as file:
        nyu2_train = list((row.split(',') for row in file.read().split('\n') if len(row) > 0))


    with open('Archive/test.csv', 'r') as file:
        nyu2_test = list((row.split(',') for row in file.read().split('\n') if len(row) > 0))


    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # For training
##    nyu2_train = nyu2_train[:10]
##    nyu2_test = nyu2_test[:10]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth

def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]


            img0 = Image.open( "Archive/" + sample[0] )
            resized_image = img0.resize((640, 480))
            resized_img0 = resized_image.convert('RGB')

            if sample[1][0] == 'd':
                exr_image = cv2.imread("Archive/" + sample[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                gray = exr_image[..., 2]
                gray *= 10
                inverted_array = 1 - gray
                scaled_array = (inverted_array * 255).astype(np.uint8)
                image = Image.fromarray(scaled_array, mode='L')
                resized_img1 = image.resize((640, 480))
            else:
                img1 = Image.open( "Archive/" + sample[1] ).convert("L")
                resized_img1 = img1.resize((640, 480))

            x = np.clip(np.asarray(resized_img0).reshape(480,640,3)/255,0,1)
            y = np.clip(np.asarray(resized_img1).reshape(480,640,1)/255*self.maxDepth,0,self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            img0 = Image.open( "Archive/" + sample[0] )
            resized_image = img0.resize((640, 480))
            resized_img0 = resized_image.convert('RGB')

            if sample[1][0] == 'd':
                exr_image = cv2.imread("Archive/" + sample[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                gray = exr_image[..., 2]
                gray *= 10
                inverted_array = 1 - gray
                scaled_array = (inverted_array * 255).astype(np.uint8)
                image = Image.fromarray(scaled_array, mode='L')
                resized_img1 = image.resize((640, 480))
            else:
                img1 = Image.open( "Archive/" + sample[1] ).convert("L")
                resized_img1 = img1.resize((640, 480))

            x = np.clip(np.asarray(resized_img0).reshape(480,640,3)/255,0,1)
            y = np.asarray(resized_img1, dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

#================
# Unreal dataset
#================

import cv2
from skimage.transform import resize

def get_unreal_data(batch_size, unreal_data_file='unreal_data.h5'):
    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Open data file
    import h5py
    data = h5py.File(unreal_data_file, 'r')

    # Shuffle
    from sklearn.utils import shuffle
    keys = shuffle(list(data['x'].keys()), random_state=0)

    # Split some validation
    unreal_train = keys[:len(keys)-100]
    unreal_test = keys[len(keys)-100:]

    # Helpful for testing...
    if False:
        unreal_train = unreal_train[:10]
        unreal_test = unreal_test[:10]

    return data, unreal_train, unreal_test, shape_rgb, shape_depth

def get_unreal_train_test_data(batch_size):
    data, unreal_train, unreal_test, shape_rgb, shape_depth = get_unreal_data(batch_size)
    
    train_generator = Unreal_BasicAugmentRGBSequence(data, unreal_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = Unreal_BasicAugmentRGBSequence(data, unreal_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, is_skip_policy=True)

    return train_generator, test_generator

class Unreal_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.N = len(self.dataset)
        self.is_skip_policy = is_skip_policy

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        
        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
            
            rgb_sample = cv2.imdecode(np.asarray(self.data['x/{}'.format(sample)]), 1)
            depth_sample = self.data['y/{}'.format(sample)] 
            depth_sample = resize(depth_sample, (self.shape_depth[1], self.shape_depth[2]), preserve_range=True, mode='reflect', anti_aliasing=True )
            
            x = np.clip(rgb_sample/255, 0, 1)
            y = np.clip(depth_sample, 10, self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])
                
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i],self.maxDepth)/self.maxDepth,0,1), index, i)

        return batch_x, batch_y
