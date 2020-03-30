import skimage.io as io
import numpy as np
import h5py
import os
import sys


class NYUDataLoader :
    def __init__(self, path):
        self.path = path
        
    def load_data(self, count = None):
        if count is None:
            count = sys.maxsize
    def load_image():
        raise NotImplementedError
    def get_image():
        raise NotImplementedError


# data path
path_to_depth = './nyu_depth_v2_labeled.mat'

# read mat file
f = h5py.File(path_to_depth)

# read 0-th image. original format is [3 x 640 x 480], uint8
img = f['images'][0]

# reshape
img_ = np.empty([480, 640, 3])
img_[:,:,0] = img[0,:,:].T
img_[:,:,1] = img[1,:,:].T
img_[:,:,2] = img[2,:,:].T


# imshow
img__ = img_.astype('float32')
io.imshow(img__/255.0)
io.show()
io.imsave('tst.png', img__)


# read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
depth = f['instances'][0]

# reshape for imshow
depth_ = np.empty([480, 640, 3])
depth_[:,:,0] = depth[:,:].T
depth_[:,:,1] = depth[:,:].T
depth_[:,:,2] = depth[:,:].T

io.imshow(depth_/4.0)
io.show()

def showImg(img):
    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T

    # imshow
    img__ = img_.astype('float32')
    io.imshow(img__/255.0)
    io.show()
    
def saveImg(img, path):
    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T

    # imshow
    img__ = img_.astype('uint8')
    io.imsave(path, img__)

def showDepth(depth):
    # reshape
    depth_ = np.empty([480, 640, 3])
    depth_[:,:,0] = depth[:,:].T
    depth_[:,:,1] = depth[:,:].T
    depth_[:,:,2] = depth[:,:].T
    io.imshow(depth_/4.0)
    io.show()

length = len(f['images'])
for i in range(length) :
    img = f['images'][i]
    # showImg(img)
    sceneType = showData('sceneTypes', i)
    rawRgbFilename = showData('rawRgbFilenames', i)
    if not os.path.exists(sceneType):
        os.makedirs(sceneType)
    name = rawRgbFilename.split("/",1)[1].split(".",1)[0] + '.png'
    path = './'+ sceneType + '/'+ name
    #saveImg(img, path)
    
    