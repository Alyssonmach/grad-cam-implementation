# importação dos pacotes
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime import lime_image
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

def get_img_array(img_path, size):
    '''retorna o array de uma imagem'''
  
    # carregando a imagem com o keras e organizando suas dimensões
    try:
      img = keras.preprocessing.image.load_img(img_path, target_size=size)
    except:
      img = img_path
    # pegando o array de píxels de cada um dos canais da imagem (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # adicionando a dimensão de 'batch' (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)

    return array

def lime_plot(path, img_size, model_base, top_labels, label_select, preprocess_input, 
              num_samples = 3000):
  '''plota os gráficos do algoritmo lime'''

  explainer = lime_image.LimeImageExplainer()
  img_path = keras.preprocessing.image.load_img(path, target_size = img_size)

  img_array = preprocess_input(get_img_array(img_path, img_size))

  explanation = explainer.explain_instance(img_array[0].astype('double'), model_base.predict, 
                                          top_labels = top_labels, hide_color=0, num_samples=3000)

  print('image predict: {} %'.format((model_base.predict(img_array) * 100)))

  plt.figure(figsize = (15, 15))

  plt.subplot(2, 2, 1)
  plt.imshow(img_array[0])

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[label_select], positive_only=True, 
                                              num_features=50, hide_rest=False)
  plt.subplot(2, 2, 2)
  plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[label_select], positive_only=False, 
                                              num_features=50, hide_rest=False)
  plt.subplot(2, 2, 3)
  plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

  ind =  explanation.top_labels[label_select]

  dict_heatmap = dict(explanation.local_exp[ind])
  heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

  plt.subplot(2, 2, 4)
  plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
  plt.colorbar()

  return heatmap 
