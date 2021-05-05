import numpy as np

def rgb_to_grayscale(np_array):
  return np.dot(np_array, [0.33, 0.33, 0.33])

def grayscale_to_rgb(np_array):
  return np.stack((np_array,)*3, axis=-1)

def add_color_channel(np_array):
  return np_array[..., None]

# veränderung 1