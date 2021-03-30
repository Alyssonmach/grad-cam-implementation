# importando os pacotes
import tensorflow as tf
from tensorflow import keras
import numpy as np
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    '''constroi o mapa de calor do grad cam'''

    # definindo a entrada e a saída do modelo com base na arquitetura importada
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [model.get_layer(last_conv_layer_name).output, 
                                       model.output])

    # calculando o gradiente da classe predita superior para a imagem de entrada
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # gradiente do neurônio de saída
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # este é um vetor onde cada entrada é a intensidade média do gradiente
    # em um canal de mapa de recurso específico
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiplicando cada canal na matriz do mapa de recursos
    # por "quão importante este canal é" em relação à melhor classe prevista
    # em seguida, some todos os canais para obter a ativação da classe do mapa de calor
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # normalizando o mapa de calor para fins de visualização
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    '''salva e plota o mapa de calor sobreposto na imagem de base'''

    # carregando a imagem original
    try:
      img = keras.preprocessing.image.load_img(img_path)
    except:
      img = img_path
    img = keras.preprocessing.image.img_to_array(img)

    # colocando o mapa de calor da escala adequada
    heatmap = np.uint8(255 * heatmap)

    # colorindo o mapa de calor
    jet = cm.get_cmap("jet")

    # usando o padrão RGB para as cores do mapa de calor
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # criando uma imagem com o mapa de calor reajustado
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # sobrepondo o mapa de calor e a imagem em um mesmo ambiente de plotagem
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # salvando a imagem obtida
    superimposed_img.save(cam_path)

    # plotando a imagem obtida
    display(Image(cam_path))
    
    return jet_heatmap
    
def get_grad_cam(img_size, preprocess_input, last_conv_layer, path, model_base, model_grad, save_file = 'image.png'):
    '''plota os gráficos do grad cam'''

    # baixando e visualizando a imagem a ser utilizada com o Grad-Cam
    img_path = keras.preprocessing.image.load_img(path, target_size = img_size)

    # preparando a imagem
    img_array = preprocess_input(get_img_array(img_path, img_size))

    # fazendo uma predição com a rede para a imagem utilizada 
    print('Porcentagem de anormalidade:', model_base.predict(img_array) * 100,'%')

    # plotando a imagem
    display(img_path)

    # removendo a função de ativação da última camada
    model_grad.layers[-1].activation = None

    # gerando o mapa de ativação de classe (Grad-Cam)
    heatmap = make_gradcam_heatmap(img_array, model_grad, last_conv_layer)

    # visualizando o mapa de calor gerado pelo Grad-Cam
    plt.matshow(heatmap)
    plt.show()

    # resultado final do algoritmo Grad-Cam
    heatmap_grad = save_and_display_gradcam(img_path, heatmap, cam_path = save_file)

    return heatmap_grad
