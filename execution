from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline


# ファイルをアップロード
uploaded = files.upload()


# アップロードされたファイルの名前を取得
uploaded_file_name = next(iter(uploaded))


# 画像を読み込みます
original_image = Image.open(uploaded_file_name)


# 画像を表示します
#plt.figure()
#plt.imshow(original_image, cmap=plt.cm.binary)
#plt.title("Original Image")
#plt.show()


# 画像を読み込んで前処理します
uploaded_image = Image.open(uploaded_file_name).convert('L')


plt.figure()
plt.imshow(uploaded_image, cmap=plt.cm.binary)
plt.title("L Image")
#plt.show()


uploaded_image = uploaded_image.resize((28, 28))
uploaded_image = np.array(uploaded_image) / 255.0
uploaded_image = uploaded_image.reshape(1, 28, 28)


# 予測を実行します
predictions = model.predict(uploaded_image)
predicted_label = np.argmax(predictions)


# 予測結果を表示します
plt.figure()
plt.imshow(uploaded_image[0], cmap=plt.cm.binary)
plt.title(f"Predicted: {predicted_label}")
plt.show()

