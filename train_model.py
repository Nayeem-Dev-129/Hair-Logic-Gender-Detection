import pandas as pd
import numpy as np
import os
import cv2

from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import (
    Sequential,
    Model
)

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# ==========================================
# LOAD UTKFACE DATASET
# ==========================================

BASE_DIR = "datasets/UTKFace"

image_paths = []
age_labels = []
gender_labels = []

for filename in tqdm(os.listdir(BASE_DIR)):

    temp = filename.split('_')

    if temp[0].isdigit():

        image_path = os.path.join(
            BASE_DIR,
            filename
        )

        image_paths.append(image_path)

        age_labels.append(int(temp[0]))

        gender_labels.append(int(temp[1]))

df = pd.DataFrame()

df['image'] = image_paths
df['age'] = age_labels
df['gender'] = gender_labels

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(images):

    features = []

    for image in tqdm(images):

        img = Image.open(image).convert('L')

        img = img.resize((128,128))

        img = np.array(img)

        features.append(img)

    features = np.array(features)

    features = features.reshape(
        len(features),
        128,
        128,
        1
    )

    return features

x = extract_features(df['image'])

x = x / 255.0

y_gender = np.array(df['gender'])

y_age = np.array(df['age'])

# ==========================================
# SPLIT DATA
# ==========================================

y_combined = np.column_stack(
    (y_gender, y_age)
)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y_combined,
    test_size=0.2,
    random_state=42
)

y_gender_train = y_train[:,0]
y_age_train = y_train[:,1]

y_gender_test = y_test[:,0]
y_age_test = y_test[:,1]

# ==========================================
# AGE + GENDER MODEL
# ==========================================

inputs = Input((128,128,1))

x1 = Conv2D(
    32,
    (3,3),
    activation='relu'
)(inputs)

x1 = MaxPooling2D((2,2))(x1)

x1 = Conv2D(
    64,
    (3,3),
    activation='relu'
)(x1)

x1 = MaxPooling2D((2,2))(x1)

x1 = Conv2D(
    128,
    (3,3),
    activation='relu'
)(x1)

x1 = MaxPooling2D((2,2))(x1)

x1 = Flatten()(x1)

dense1 = Dense(
    256,
    activation='relu'
)(x1)

dense2 = Dense(
    256,
    activation='relu'
)(x1)

drop1 = Dropout(0.3)(dense1)

drop2 = Dropout(0.3)(dense2)

gender_output = Dense(
    1,
    activation='sigmoid',
    name='gender_out'
)(drop1)

age_output = Dense(
    1,
    activation='relu',
    name='age_out'
)(drop2)

gender_age_model = Model(
    inputs=inputs,
    outputs=[gender_output, age_output]
)

gender_age_model.compile(
    optimizer='adam',
    loss=['binary_crossentropy', 'mae'],
    metrics=['accuracy']
)

print("\nTraining Age + Gender Model...\n")

gender_age_model.fit(
    x_train,
    [y_gender_train, y_age_train],
    epochs=10,
    batch_size=64,
    validation_data=(
        x_test,
        [y_gender_test, y_age_test]
    )
)

# ==========================================
# SAVE AGE + GENDER MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

gender_age_model.save(
    "models/best_model.keras"
)

print("\nAge + Gender Model Saved\n")

# ==========================================
# LOAD HAIR DATASET
# ==========================================

hair_paths = []
hair_labels = []

HAIR_DIR = "datasets/hair_dataset"

for label, folder in enumerate(
    ['long_hair', 'short_hair']
):

    folder_path = os.path.join(
        HAIR_DIR,
        folder
    )

    for file in os.listdir(folder_path):

        path = os.path.join(
            folder_path,
            file
        )

        hair_paths.append(path)

        hair_labels.append(label)

hair_df = pd.DataFrame()

hair_df['image'] = hair_paths
hair_df['label'] = hair_labels

# ==========================================
# HAIR FEATURE EXTRACTION
# ==========================================

def extract_hair_features(images):

    features = []

    for image in tqdm(images):

        img = Image.open(image).convert('RGB')

        img = img.resize((128,128))

        img = np.array(img)

        features.append(img)

    return np.array(features)

x_hair = extract_hair_features(
    hair_df['image']
)

x_hair = x_hair / 255.0

y_hair = np.array(
    hair_df['label']
)

# ==========================================
# HAIR MODEL
# ==========================================

hair_model = Sequential()

hair_model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)

hair_model.add(
    MaxPooling2D((2,2))
)

hair_model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

hair_model.add(
    MaxPooling2D((2,2))
)

hair_model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

hair_model.add(
    MaxPooling2D((2,2))
)

hair_model.add(
    Flatten()
)

hair_model.add(
    Dense(
        128,
        activation='relu'
    )
)

hair_model.add(
    Dropout(0.3)
)

hair_model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

hair_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Hair Model...\n")

hair_model.fit(
    x_hair,
    y_hair,
    epochs=10,
    batch_size=32
)

# ==========================================
# SAVE HAIR MODEL
# ==========================================

hair_model.save(
    "models/hair_model.keras"
)

print("\nHair Model Saved\n")