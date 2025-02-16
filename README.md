# DL_Classification_CIFAR

# Classification d'Images avec CNN sur CIFAR-10

Ce projet implémente un **CNN** pour classer des images du dataset **CIFAR-10** en 10 classes :  
`['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`.

---

##  1. Installation

Exécute les commandes suivantes :

```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

---

##  2. Prétraitement des Données  

- **Chargement et division** en train/validation/test  
- **Normalisation** avec la moyenne et l’écart type  
- **Data augmentation** : rotation, zoom, flip, etc.

```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-7)
```

---

## 3. Modèle CNN  

Architecture avec **BatchNorm, MaxPooling, Dropout** :

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    BatchNormalization(), MaxPooling2D((2,2)), Dropout(0.2),
    Flatten(), Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

##  4. Entraînement  

- **50 époques**, batch size **64**  
- **Réduction du learning rate** et **EarlyStopping**  

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

model.fit(data_generator.flow(X_train, y_train, batch_size=64),
          epochs=50, validation_data=(X_valid, y_valid),
          callbacks=[reduce_lr, early_stopping], verbose=2)
```

---

## 5. Évaluation et Prédiction  

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
image = cv2.imread("image_path.jpeg")
image = cv2.resize(image, (32,32))
image = (image - X_train.mean()) / (X_train.std() + 1e-7)
prediction = model.predict(image.reshape(1, 32, 32, 3))
print('Predicted class:', class_names[prediction.argmax()])
```

---

## 6. Sauvegarde du Modèle  

```python
model.save('/content/drive/MyDrive/data/moncifar.h5')
model.save_weights('/content/drive/MyDrive/data/poids_moncifar.h5')
```

---

## 7. Améliorations Possibles  

🔹 Tester **ResNet/VGG** (Transfer Learning)  
🔹 Ajouter **plus d'images pour l'entraînement**  
🔹 Optimiser l'architecture avec **régularisation**  


