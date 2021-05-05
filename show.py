import math
from matplotlib import pyplot as plt

def show_images_from_np_array(np_array, title, width=8):
  fig = plt.figure(figsize=(width, width))
  x_size = np_array.shape[0]
  x_size_sqrt = math.ceil(math.sqrt(x_size))
  if (len(np_array.shape)) == 3:
    cmap = "gray"
  else:
    cmap = "viridis"

  for x in range(x_size):
    plt.subplot(x_size_sqrt, x_size_sqrt, x+1)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np_array[x], cmap=cmap)
  
  fig.suptitle(title, fontsize=16)


def show_batch(batch, width=8):
  show_images_from_np_array(batch[0], "Batch X", width)
  show_images_from_np_array(batch[1], "Batch Y", width)

def show_and_compare_batch(batch, width=4):
  batch_size = batch[0].shape[0]
  fig = plt.figure(figsize=(width, width/2*batch_size))
  for x in range(batch_size):
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

def show_frame_prediction_batch(batch):
  n_rows = batch[0].shape[0]
  n_columns = batch[0][0].shape[0]+1

  fig = plt.figure(figsize=(n_rows, n_columns))
  
  for row in range(n_rows):
    for column in range(n_columns-1):
      subplot_idx = (n_columns * row) + column + 1
      plt.subplot(n_rows, n_columns, subplot_idx)
      plt.xticks(())
      plt.yticks(())
      img = batch[0][row][column]
      plt.imshow(img, cmap="gray")
    subplot_idx = (n_columns * row) + n_rows + 1
    plt.subplot(n_rows, n_columns, subplot_idx)
    img = batch[1][row]
    plt.imshow(img, cmap="gray")
    plt.title("y")
    plt.xticks(())
    plt.yticks(())