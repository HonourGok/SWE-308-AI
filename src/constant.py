import random

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 1
LEARNING_RATE = 0.0001
MODEL_PATH = os.getcwd()+"\\model\\gender_classifier_resnet50v2.h5"


train_dir = os.getcwd()+"\\dataset\\train"
validation_dir = os.getcwd()+"\\dataset\\validation"
test_dir = os.getcwd()+"\\dataset\\test"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['MEN', 'WOMAN']
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['MEN', 'WOMAN']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode=None,
    shuffle=True
)

# Model varsa yükle, yoksa oluştur ve eğit
if os.path.exists(MODEL_PATH):
    print(" Eğitilmiş model yüklendi.")
    model = load_model(MODEL_PATH)
else:
    print("Yeni model eğitiliyor...")
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Extract accuracy and loss values
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(train_acc) + 1)

    # Plot accuracy graph
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iteration")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Iteration")
    plt.legend()
    plt.grid()
    plt.show()

    # Eğitilen modeli kaydet
    model.save(MODEL_PATH)
    print("Model kaydedildi.")


def show_classification_results(y_true, y_pred, class_labels=['MEN', 'WOMAN']):
    report = classification_report(y_true, y_pred, target_names=["MEN", "WOMAN"])
    print("\n📊 Performance Metrics:\n")
    print(report)

    # Karmaşıklık matrisi
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Karmaşıklık Matrisi:")
    print(conf_matrix)

    # Görsel olarak çizim
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Tek görsel için tahmin
def predict_gender(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    gender = 'MEN' if prediction < 0.5 else 'WOMAN'
    return gender

# Test görselleri için tahmin
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg'))]
# for image in test_images:
#     print(f"Görsel: {image}, Tahmin Edilen Cinsiyet: {predict_gender(image)}")


y_true = validation_generator.classes

# Generate predictions
y_pred = model.predict(validation_generator)
y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary class labels

show_classification_results(y_true, y_pred)


def show_random_validation_images(model, val_generator, class_names, image_size=IMAGE_SIZE):
    """Pick 3 random validation images, predict their labels, and display true vs. predicted."""
    # Select 3 random indices from validation dataset
    random_indices = random.sample(range(len(val_generator.filenames)), 3)

    plt.figure(figsize=(10, 4))

    for i, idx in enumerate(random_indices):
        # Load image & true label
        img_path = os.path.join(validation_dir, val_generator.filenames[idx])  # Full path
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        true_label = val_generator.classes[idx]  # Get true class index
        true_label_str = class_names[true_label]  # Convert index to class name

        # Predict
        prediction = model.predict(img_array)
        predicted_label = "MEN" if prediction < 0.5 else "WOMAN"

        # Display image with true and predicted labels
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label_str}, Predicted: {predicted_label}")
        plt.axis("off")

    plt.show()


# Get class names from training dataset
class_names = list(train_generator.class_indices.keys())  # ✅ Converts to a list

# Call function to display 3 random images with true and predicted labels
show_random_validation_images(model, validation_generator, class_names)

print(model.summary())