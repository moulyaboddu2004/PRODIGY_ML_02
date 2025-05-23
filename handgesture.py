import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Path to dataset
DATASET_PATH = r"C:\Users\boddu\Downloads\leapGestData2\leapGestRecog"

# Rename files without extensions recursively
for root, dirs, files in os.walk(DATASET_PATH):
    for filename in files:
        if '.' not in filename:
            old_path = os.path.join(root, filename)
            new_path = old_path + '.png'  # or '.jpg' if your images are JPGs
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")

# Parameters
IMG_SIZE = 100
EPOCHS = 15
BATCH_SIZE = 32

# Load data
images = []
labels = []
label_map = {}
label_id = 0

print("Loading images...")

for user in sorted(os.listdir(DATASET_PATH)):
    user_path = os.path.join(DATASET_PATH, user)
    if not os.path.isdir(user_path):
        continue

    for gesture in sorted(os.listdir(user_path)):
        gesture_path = os.path.join(user_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        if gesture not in label_map:
            label_map[gesture] = label_id
            label_id += 1

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_map[gesture])

print("Data loaded successfully.")

# Preprocess data
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
labels = to_categorical(np.array(labels), num_classes=len(label_map))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[early_stop])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2f}\n")

# Confusion Matrix & Classification Report
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:\n")
print(classification_report(y_true_labels, y_pred_labels))

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
