# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.tests import test_image

#자동으로 다운로드
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

print(train_images.shape)

#1차원으로 변형
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

print(train_images.shape)

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer='adam', 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model

#모델 저장 패스
checkpoint_path = "model_data/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback (콜백으로 계속 가중치를 기록)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 2, 
          validation_data = (train_images,train_labels),
          callbacks = [cp_callback])  # pass callback to training

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#저장된 모델 가중치 불러오는 부분
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#print(model.get_weights())


predictions = model.predict(test_images)
print(predictions[2])

print(np.argmax(predictions[2]))


 
img = test_images[2].reshape(28,28)

plt.imshow(img, cmap=plt.get_cmap("binary"))
#plot_value_array(i, predictions,  test_labels)
plt.show()
#print(test_images[2])

