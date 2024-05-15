{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2a318185-120c-49c8-939b-30b72e687248",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'datasets' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[3], line 19\u001b[0m\n\u001b[0;32m     16\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m np\u001b[38;5;241m.\u001b[39marray(images)\n\u001b[0;32m     18\u001b[0m \u001b[38;5;66;03m# Load car images from CIFAR-10 dataset\u001b[39;00m\n\u001b[1;32m---> 19\u001b[0m cifar_dir \u001b[38;5;241m=\u001b[39m datasets\u001b[38;5;241m.\u001b[39mcifar10\u001b[38;5;241m.\u001b[39mload_data()\n\u001b[0;32m     20\u001b[0m (x_train, y_train), (_, _) \u001b[38;5;241m=\u001b[39m tf\u001b[38;5;241m.\u001b[39mkeras\u001b[38;5;241m.\u001b[39mdatasets\u001b[38;5;241m.\u001b[39mcifar10\u001b[38;5;241m.\u001b[39mload_data()\n\u001b[0;32m     21\u001b[0m car_images \u001b[38;5;241m=\u001b[39m x_train[np\u001b[38;5;241m.\u001b[39mwhere(y_train \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m)[\u001b[38;5;241m0\u001b[39m]][:\u001b[38;5;241m500\u001b[39m]  \u001b[38;5;66;03m# Assuming car class is labeled as 1\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'datasets' is not defined"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import layers, models\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from sklearn.model_selection import train_test_split\n",
    "from PIL import Image\n",
    "\n",
    "def load_and_preprocess_images(file_paths, target_size=(32, 32)):\n",
    "    images = []\n",
    "    for path in file_paths:\n",
    "        image = Image.open(path)\n",
    "        image = image.resize(target_size)\n",
    "        image = np.array(image) / 255.0  # Normalize pixel values\n",
    "        images.append(image)\n",
    "    return np.array(images)\n",
    "\n",
    "cifar_dir = datasets.cifar10.load_data()\n",
    "(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()\n",
    "car_images = x_train[np.where(y_train == 1)[0]][:500]\n",
    "\n",
    "gun_paths = ['gun1.jpg', 'gun2.jpg', 'gun3.jpg', 'gun4.jpg']\n",
    "gun_images = load_and_preprocess_images(gun_paths)\n",
    "\n",
    "X = np.concatenate([car_images, gun_images])\n",
    "y = np.concatenate([np.ones(len(car_images)), np.zeros(len(gun_images))])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "model = models.Sequential([\n",
    "    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Conv2D(128, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D((2, 2)),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(128, activation='relu'),\n",
    "    layers.Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam',\n",
    "              loss='binary_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rotation_range=20,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest')\n",
    "\n",
    "model.fit(train_datagen.flow(X_train, y_train, batch_size=32), \n",
    "          steps_per_epoch=len(X_train) // 32, epochs=10,\n",
    "          validation_data=(X_test, y_test))\n",
    "\n",
    "loss, accuracy = model.evaluate(X_test, y_test)\n",
    "print(f\"Test Accuracy: {accuracy}\")\n",
    "\n",
    "model.save(\"car_gun_classifier_model.h5\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0b28efc-1861-427d-90ca-de14965309ca",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
