import math
from matplotlib import pyplot as plt

def show_images_from_np_array(np_array, title, width=8, img_limit=999999):
  fig = plt.figure(figsize=(width, width))
  n_images = np_array.shape[0]
  n_images_sqrt = math.ceil(math.sqrt(n_images))
  if (len(np_array.shape)) == 3:
    cmap = "gray"
  else:
    cmap = "viridis"
  
  img_limit = min(n_images, img_limit)
  for x in range(img_limit):
    plt.subplot(n_images_sqrt, n_images_sqrt, x+1)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np_array[x], cmap=cmap, vmin=0, vmax=1)
  
  fig.suptitle(title, fontsize=16)


def show_batch(batch, width=8, img_limit=999999):
  show_images_from_np_array(batch[0], "Batch X", width, img_limit)
  show_images_from_np_array(batch[1], "Batch Y", width, img_limit)

def show_and_compare_batch(batch, width=4, img_limit=99999):
  batch_size = batch[0].shape[0]
  fig = plt.figure(figsize=(width, width/2*batch_size))
  img_limit = min(img_limit, batch_size)
  for x in range(img_limit):
    plt.subplot(batch_size, 2, 2 * x+1)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(batch[0][x])
    plt.title("x")
    plt.subplot(batch_size, 2, 2 * x+2)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(batch[1][x])
    plt.title("y")

def show_frame_prediction_batch(batch, row_limit=999999):
  n_rows = batch[0].shape[0]
  n_columns = batch[0][0].shape[0]+1

  fig = plt.figure(figsize=(n_columns, n_rows*1.5))
  
  row_limit = min(row_limit, n_rows)
  for row in range(row_limit):
    for column in range(n_columns-1):
      subplot_idx = (n_columns * row) + column + 1
      plt.subplot(n_rows+1, n_columns, subplot_idx)
      plt.xticks(())
      plt.yticks(())
      img = batch[0][row][column]
      plt.imshow(img, cmap="gray")
    subplot_idx = (n_columns * (row)) + n_columns
    plt.subplot(n_rows+1, n_columns, subplot_idx)
    img = batch[1][row]
    plt.imshow(img, cmap="gray")
    plt.title("y")
    plt.xticks(())
    plt.yticks(())

# veränderung 1

# veränderung 2
