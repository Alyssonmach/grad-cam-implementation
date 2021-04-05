#importação dos pacotes 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")


def pre_processamento(test_data = True):
    
    # lendo os dados de um arquivo csv
    dataframe = pd.read_csv('/content/drive/MyDrive/vinbigdata/train.csv')
    # criando uma coluna com os caminhos relativos as imagens
    dataframe['image_path'] = '/content/drive/MyDrive/vinbigdata/train/' + dataframe.image_id + '.jpg'

    print('total de imagens disponíveis:', str(len(set(dataframe['image_path']))))

    # visualizando os casos disponíveis
    dataframe['class_name'].value_counts()

    # removendo os casos não relativos a distúrbios pulmonares
    dataframe = dataframe[dataframe.class_name != 'Aortic enlargement']
    dataframe = dataframe[dataframe.class_name != 'Cardiomegaly']
    dataframe = dataframe[dataframe.class_name != 'Other lesion']
    dataframe = dataframe[dataframe.class_name != 'Consolidation']

    # separando os casos rotulados como normais e anormais
    normal_cases = dataframe[(dataframe.class_id == 14) & (dataframe.class_name == 'No finding')]
    abnormal_cases = dataframe[(dataframe.class_id != 14) & (dataframe.class_name != 'No finding')]

    print('\ntotal de dados após a filtração:', str(len(set(normal_cases['image_path'])) + len(set(abnormal_cases['image_path']))))

    # removendo as imagens repetidas
    normal_path = list(set(normal_cases['image_path']))
    abnormal_path = list(set(abnormal_cases['image_path']))

    # criando dataframes especifos com caminhos para as imagens e rótulos
    normal_data = pd.DataFrame(normal_path, columns = ['filepath'])
    normal_data['target'] = 0
    abnormal_data = pd.DataFrame(abnormal_path, columns = ['filepath'])
    abnormal_data['target'] = 1

    print('\nquantidade de dados rotulados como normais:', len(normal_data))
    print('quantidade de dados rotulados como anormais:', len(abnormal_data))

    # removendo 69% dos casos normais para balancear os dados
    normal, _ = train_test_split(normal_data, test_size = 0.69, random_state = 42)

    print('\nquantidade de dados rotulados como normais (balanceado):', len(normal))
    print('quantidade de dados rotulados como anormais:', len(abnormal_data))

    # concatenando os dataframes de casos normais e anormais
    full_data = pd.concat([normal, abnormal_data])

    # misturando todos os dados do dataframe e reiniciando os valores dos índices 
    full_data = full_data.sample(frac = 1, axis = 0, random_state = 42).reset_index(drop=True)

    # modificando o formato dos dados para float32
    dict_type = {'target': 'float32'}
    full_data = full_data.astype(dict_type)

    # separando os dados de treinamento e de teste
    train_df, test_df = train_test_split(full_data, stratify = full_data['target'],
                                         test_size = 0.2, random_state = 42)

    # separando os dados de validação dos dados de treinamento
    train_df, validation_df = train_test_split(train_df, stratify = train_df['target'],
                                               test_size = 0.2, random_state = 42)

    # visualizando a quantidade de dados
    print('\nquantidade de imagens de treinamento:', len(train_df['filepath']))
    print('quantidade de rótulos de treinamento:', len(train_df['target']))
    print('quantidade de imagens de teste:', len(test_df['filepath']))
    print('quantidade de rótulos de teste:', len(test_df['target']))
    print('quantidade de imagens de validação:', len(validation_df['filepath']))
    print('quantidade de rótulos de validação:', len(validation_df['target']), '\n')

    # normalizando as imagens de treinamento e aplicando aumento de dados
    image_generator = ImageDataGenerator(rescale = 1./255.,
                                         rotation_range = 10, zoom_range = 0.2)

    # criando o gerador de imagens de treinamento 
    train_generator = image_generator.flow_from_dataframe(
                                                          dataframe = train_df,
                                                          directory = '',
                                                          x_col = 'filepath',
                                                          y_col = 'target',
                                                          batch_size = 32,
                                                          seed = 42,
                                                          shuffle = True,
                                                          class_mode = 'raw',
                                                          target_size = (256, 256))
    # criando o gerador de imagens de validação 
    valid_generator = image_generator.flow_from_dataframe(
                                                          dataframe = validation_df,
                                                          directory = '.', 
                                                          x_col = 'filepath',
                                                          y_col = 'target',
                                                          batch_size = 32,
                                                          seed = 42,
                                                          shuffle = True,
                                                          class_mode = 'raw',
                                                          target_size = (256, 256))

    # normalizando as imagens de teste 
    test_datagen = ImageDataGenerator(rescale = 1./255.)

    test_generator = test_datagen.flow_from_dataframe(
                                                      dataframe = test_df, 
                                                      directory = '.',
                                                      x_col = 'filepath',
                                                      y_col = 'target',
                                                      batch_size = 32,
                                                      seed = 42,
                                                      shuffle = True,
                                                      class_mode = 'raw',
                                                      target_size = (256, 256))

    # carregando o melhor modelo para realização de testes de desempenho
    model_grad = tf.keras.models.load_model('/content/drive/MyDrive/experimentos/experimento2-dataset4/model2')
    model_base = tf.keras.models.load_model('/content/drive/MyDrive/experimentos/experimento2-dataset4/model2')
    
    if test_data == True:
        # carregando os dados de teste
        for i in range(0, 42):
          (x1, y1) = test_generator[i]
          if i == 0:
            x, y = x1, y1
          else:
            x = np.concatenate((x, x1))
            y = np.concatenate((y, y1))
    else:
        x, y = list(), list()
    return model_grad, model_base, (x, y)
