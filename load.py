from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from PIL import Image
import numpy as np
import os
import getpass

def load_and_unzip_from_url(zipurl):
  with urlopen(zipurl) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
      zfile.extractall()

def img_path_to_np_array(path):
  img = Image.open(path)
  img_as_array = np.array(img)
  return (img_as_array / 255.0).astype("float32")

def all_abs_paths_in_dir(dir_):
  filenames = os.listdir(dir_)
  return [os.path.join(dir_, filename) for filename in filenames]

# veränderung!!!