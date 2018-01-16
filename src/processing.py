import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction

def process(df, func):
    def work_on_bands(bands):
        return np.array([func(band) for band in bands])[np.newaxis]
    array = np.concatenate([work_on_bands(bands) for bands in df], axis=0)
    merged = np.concatenate([df, array], axis=1)
    return np.moveaxis(merged, 1, 3)

# Isolation function.
def isolate(arr):
    image = np.reshape(np.array(arr), [75,75])
#     image = img_as_float(np.reshape(np.array(arr), [75,75]))
    image = gaussian_filter(image, 2)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image 
    dilated = reconstruction(seed, mask, method='dilation')
    return image-dilated
