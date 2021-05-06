from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
import os
import random

from ml_utils import load
from ml_utils import convert


class FramePredictionGenerator(Sequence):

  def __init__(self, dir_, frame_shape, n_frames, batch_size=50, validation_fraction=0.1):
    random.seed(0)

    self.frame_shape = frame_shape
    self.n_frames = n_frames
    self.batch_size = batch_size
    
    # Pfade zu allen Verzeichnissen, die Frames einer Szene enthalten:
    scene_dir_paths =  load.all_abs_paths_in_dir(dir_)
    
    # all_examples soll die PFADE zu allen Beispielen speichern. Wenn n_frames = 3, dann so:
    # [[frame0, frame1, frame2, frame3],
    #  [frame1, frame2, frame3, frame4],
    #  [frame2, frame3, frame4, frame5],
    #  ...
    self.all_examples = []
    
    # für jede Szene:
    for scene_dir_path in scene_dir_paths:
      # Pfade zu allen Dateien:
      all_frame_paths_in_scene = load.all_abs_paths_in_dir(scene_dir_path)
      list.sort(all_frame_paths_in_scene)

      # Wie viele Beispiele können aus einer Szene gewonnen werden?
      number_of_frames = len(all_frame_paths_in_scene)
      number_of_examples = number_of_frames - self.n_frames

      # Erzeugung der Beispiele:
      for example_index in range(number_of_examples):
        example = []
        for frame_index in range(self.n_frames+1):
          # Berechnung des Indices des aktuellen Beispiels:
          index = example_index + frame_index
          # Befüllung des Beipiels:
          example.append(all_frame_paths_in_scene[index])
        # Beispiel anhängen
        self.all_examples.append(example)
      
      # Beispiele mischen, ansonsten stehen sie in der vorgegebenen Reihenfolge in der Liste:
      random.shuffle(self.all_examples) 

      # Der Index der Stelle, die Trainings- und Validierungsbeispiele trennt:
      split_index = int(len(self.all_examples) * 1.0 - validation_fraction)

      # Trennung der Trainings- von den Validierungsbeispielen:
      self.train_examples = self.all_examples[:split_index]
      self.validation_examples = self.all_examples[split_index:]

      # Ermittlung der Länge des Generators, sprich:
      # Wie viele Batches kann der Generator pro Epoche liefern?
      self.len = int(len(self.all_examples)/self.batch_size)

  def __len__(self):
    return self.len

  def __getitem__(self, batch_index):
    if batch_index >= self.len:
      raise IndexError("batch index out of range")
    else:
      # Wir erzeugen die Numpy-Arrays, welche die x und y Batches repräsentieren:
      batch_x, batch_y = self._get_empty_batches()
      # Indices der Beispiele, die in den aktuellen Batch hineingeschrieben werden sollen:
      start = int(batch_index * self.batch_size)
      stop = int((batch_index + 1) * self.batch_size)
      # Auswahl der Beispiele:
      selection = self.train_examples[start:stop]
      
      # Ausgehend von den Pfaden in den ausgewählten Beispielen...
      for example_idx, example in enumerate(selection):
        # Befüllen von batch_x:
        for frame_idx, img_path in enumerate(example[:-1]):
          rgb = load.img_path_to_np_array(img_path)
          grayscale = convert.rgb_to_grayscale(rgb)
          reshaped = grayscale.reshape((self.frame_shape))
          batch_x[example_idx, frame_idx] = reshaped
        # Befüllen von batch_y:
        img_path = example[-1]
        rgb = load.img_path_to_np_array(img_path)
        grayscale = convert.rgb_to_grayscale(rgb)
        reshaped = grayscale.reshape((self.frame_shape))
        batch_y[example_idx] = reshaped

    return batch_x, batch_y

  # on_epoch_end wird automatisch nach jeder Epoche aufgerufen und mischt die Beispiele:
  def on_epoch_end(self):
    random.shuffle(self.train_examples)

  # Hilfsfunktion – Erzeugung leerer Batches:
  def _get_empty_batches(self):
    empty_batch_x = np.empty((self.batch_size,) + (self.n_frames,) + (self.frame_shape), "float32")
    empty_batch_y = np.empty((self.batch_size,) + (self.frame_shape), "float32")
    return empty_batch_x, empty_batch_y

  # Liefert die Daten, um einen Validierungs-Generator zu bauen:
  def get_validation_data(self):
    return self.validation_examples, self.frame_shape, self.batch_size