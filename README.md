# Hair Logic Gender Detection System

## Overview

This project is developed using:

- Deep Learning
- Computer Vision
- CNN Models
- Python GUI

The system predicts:

- Age
- Gender
- Hair Type

Project Logic:

- If age is between 20–30:
  - Long Hair → Male
  - Short Hair → Female

- Otherwise:
  - Original Gender Prediction is used

---

# Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- Tkinter
- NumPy
- Pandas

---

# Project Structure

```bash
Hair-Logic-Gender-Detection/
│
├── datasets/
│   ├── UTKFace/
│   └── hair_dataset/
│       ├── long_hair/
│       └── short_hair/
├── train_model.py
├── predict.py
├── gui.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset Used

- UTKFace Dataset
- Custom Hair Dataset

---

# Features

- Age Detection
- Gender Detection
- Hair Classification
- GUI Interface
- Conditional Logic System

---

# How To Run

## Install Libraries

```bash
pip install -r requirements.txt
```

---

## Download Datasets

---

## Train Models

```bash
python train_model.py
```

---

## Run GUI

```bash
python gui.py
```

---

# Output Example

Age: 24

Detected Gender: Male

Hair Type: Long

---

# Author

Mohammed Nayeem Uddin
