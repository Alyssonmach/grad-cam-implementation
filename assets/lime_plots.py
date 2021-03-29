# importação dos pacotes
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime import lime_image
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

def lime_plot(path, img_size, model_base):
  '''plota os gráficos do algoritmo lime'''

  explainer = lime_image.LimeImageExplainer()
  img_path = keras.preprocessing.image.load_img(path, target_size = img_size)

  img_array = preprocess_input(get_img_array(img_path, img_size))

  explanation = explainer.explain_instance(img_array[0].astype('double'), model_base.predict, 
                                          top_labels = 1, hide_color=0, num_samples=3000)

  print('image predict: {}'.format(float(model_base.predict(img_array) * 100)))

  plt.figure(figsize = (15, 15))

  plt.subplot(2, 2, 1)
  plt.imshow(img_array[0])

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, 
                                              num_features=50, hide_rest=False)
  plt.subplot(2, 2, 2)
  plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, 
                                              num_features=50, hide_rest=False)
  plt.subplot(2, 2, 3)
  plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

  ind =  explanation.top_labels[0]

  dict_heatmap = dict(explanation.local_exp[ind])
  heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

  plt.subplot(2, 2, 4)
  plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
  plt.colorbar()

  return heatmap 
