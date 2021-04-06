import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from IPython.display import Image, display

def lime_positive_areas(heatmap):
  '''função para manter apenas as áreas positivas da classe'''

  new_map = heatmap.copy(order = 'C')
  ix, iy, iz = 0, 0, 0
  for i in new_map:
    ix += 1
    for j in i:
      iy += 1
      if j <= -0:
        new_map[ix - 1, iy - 1] = 0.0
    iy = 0

  return new_map

def grad_one_channel(heatmap, image_size, threshold = 150):
  '''monta o mapa de calor da grad cam em um só canal'''

  new_map = heatmap.copy(order = 'C')
  
  channel1 = (new_map[:, :, 0] > threshold)
  channel2 = (new_map[:, :, 1] > threshold)
  channel3 = (new_map[:, :, 2] > threshold)
  one_channel = np.ones((image_size[0], image_size[1]))
  for i in range(0, image_size[0]):
    for j in range(0, image_size[1]):
      one_channel[i,j] = channel1[i,j] or channel2[i,j] or channel3[i,j]

  return one_channel, (channel1, channel2, channel3)

def heatmap_intersection(heatmap_lime, heatmap_grad, image_size):
  '''função que faz a intersecção entre os mapas de calor'''

  lime_map = heatmap_lime.copy(order = 'C')
  grad_map = heatmap_grad.copy(order = 'C')

  new_map = np.ones((image_size[0], image_size[1]))
  for i in range(0, image_size[0]):
    for j in range(0, image_size[1]):
      new_map[i,j] = lime_map[i,j] and grad_map[i,j]

  return new_map

def heatmap_difference(heatmap_lime, heatmap_grad, image_size):
  '''função que faz a intersecção entre os mapas de calor'''

  lime_map = heatmap_lime.copy(order = 'C')
  grad_map = heatmap_grad.copy(order = 'C')

  new_map = np.ones((image_size[0], image_size[1]))
  for i in range(0, image_size[0]):
    for j in range(0, image_size[1]):
      new_map[i,j] = bool(lime_map[i,j]) != bool(grad_map[i,j])

  return new_map

def heatmap_three_channels(heatmap):
  '''adiciona três canais ao mapa de calor'''

  new_map = heatmap.copy(order = 'C')
  rgb_heatmap = np.expand_dims(new_map, axis=2)
  rgb_heatmap = np.concatenate((rgb_heatmap, rgb_heatmap, rgb_heatmap), axis = 2)

  return rgb_heatmap

def display_heatmap(img_path, heatmap, image_size, cam_path = 'image.png', alpha = 255):
  '''plota o mapa de calor sobreposto na imagem'''

  try:
      img = keras.preprocessing.image.load_img(img_path, target_size = image_size)
  except:
      img = img_path

  new_map = heatmap.copy(order = 'C')
  img = keras.preprocessing.image.img_to_array(img)
  superimposed_img = new_map * alpha + img
  superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
  # salvando a imagem obtida
  superimposed_img.save(cam_path)

  # plotando a imagem obtida
  display(Image(cam_path))

  return None
