
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# MNISTデータセットのロード
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# データの前処理
train_images = train_images / 255.0
test_images = test_images / 255.0


train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)


# ニューラルネットワークモデルの構築
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


# モデルのコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 学習曲線用のデータを保存するリストを作成
train_loss_history = []
val_loss_history = []


# モデルのトレーニング
epochs = 5
for epoch in range(epochs):
    history = model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_data=(test_images, test_labels))


    # 各エポックの訓練データと検証データの誤差を保存
    train_loss_history.append(history.history['loss'][0])
    val_loss_history.append(history.history['val_loss'][0])


# モデルの性能評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')


# モデルを使用して文字認識を行う例
predictions = model.predict(test_images)
random_index = np.random.randint(0, test_images.shape[0])
predicted_label = np.argmax(predictions[random_index])
true_label = np.argmax(test_labels[random_index])


# 学習曲線をプロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_loss_history, label='Training Loss')
plt.plot(range(1, epochs+1), val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# 画像表示
plt.subplot(1, 2, 2)
plt.imshow(test_images[random_index], cmap=plt.cm.binary)
plt.title(f"Predicted: {predicted_label}, True: {true_label}")
plt.show()



