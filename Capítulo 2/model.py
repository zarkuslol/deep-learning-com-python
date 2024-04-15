# Importações
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# Carregando o dataset MNIST do Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Crio a rede e alimento-a com train_images e train_labels
rede = models.Sequential()
rede.add(layers.Dense(512, activation='relu', input_shape=(784,))) # Camada de entrada
rede.add(layers.Dense(10, activation='softmax')) # Camada de saída

rede.compile(optimizer='Adam', # rmsprop
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# - pré-processamento das imagens
train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255

# - pré-processamento dos labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

rede.fit(train_images, train_labels, epochs=10, batch_size=128)

# Jogo test_images para a rede e comparo as predições de test_images com test_labels
test_loss, test_acc = rede.evaluate(test_images, test_labels)
print(f'test_acc = {test_acc}')
