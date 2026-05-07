import cv2
import numpy as np

from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODELS
# ==========================================

gender_age_model = load_model(
    'models/best_model.keras'
)

hair_model = load_model(
    'models/hair_model.keras'
)

# ==========================================
# LABELS
# ==========================================

gender_dict = {
    0: 'Male',
    1: 'Female'
}

# ==========================================
# HAIR PREDICTION
# ==========================================

def predict_hair(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(
        img,
        (128,128)
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    pred = hair_model.predict(img)

    # 0 = Long Hair
    # 1 = Short Hair

    if pred[0][0] > 0.5:
        return "Short"

    return "Long"

# ==========================================
# MAIN PREDICTION
# ==========================================

def predict_image(image_path):

    # -----------------------------
    # AGE + GENDER PREDICTION
    # -----------------------------

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(
        img,
        (128,128)
    )

    img = img / 255.0

    img = img.reshape(
        1,
        128,
        128,
        1
    )

    pred = gender_age_model.predict(img)

    gender = gender_dict[
        int(round(pred[0][0][0]))
    ]

    age = round(pred[1][0][0])

    # -----------------------------
    # HAIR PREDICTION
    # -----------------------------

    hair_type = predict_hair(
        image_path
    )

    # -----------------------------
    # FINAL PROJECT LOGIC
    # -----------------------------

    if 20 <= age <= 30:

        if hair_type == "Long":

            final_prediction = "Female"

        else:

            final_prediction = "Male"

    else:

        final_prediction = gender

    # -----------------------------
    # RETURN RESULTS
    # -----------------------------

    return {

        "Age": age,

        "Detected Gender": gender,

        "Hair Type": hair_type,

        "Final Prediction": final_prediction
    }