# EN Special Term Project : Automatic Trash Sorter
-> Group 2   
-> Supervised by Dr.Tharaka Smarasinghe


🧭 AI Trash Classification with Raspberry Pi — Full Roadmap
⚙️ Stage 1 — Core Foundations (Week 1–2)

Goal: Learn the essential tools — Python, OpenCV, and Raspberry Pi basics.

📘 Learn

Python fundamentals: data types, loops, NumPy, file I/O

OpenCV basics: image loading, color detection, contour detection

Raspberry Pi setup: OS installation, SSH, camera setup

🧩 Practice

Use cv2.VideoCapture(0) to display live video.

Write scripts to:

Convert images to grayscale

Detect color regions (plastic often has shiny color)

Capture & save images when a button is pressed

🎓 Resources

Python Crash Course (freeCodeCamp)

OpenCV Python Course

Getting Started with Raspberry Pi Camera

🤖 Stage 2 — Machine Learning & Computer Vision Concepts (Week 2–3)

Goal: Understand how image classification models work.

📘 Learn

What is machine learning vs. deep learning

How CNNs (Convolutional Neural Networks) process images

Dataset → Training → Testing → Model evaluation

🧩 Practice

Train a simple classifier on your laptop using scikit-learn or TensorFlow.

Try classifying MNIST digits or CIFAR-10 images.

Visualize CNN layers using TensorBoard.

🎓 Resources

Deep Learning Crash Course – freeCodeCamp

Google ML Crash Course

Kaggle: Intro to Machine Learning

📸 Stage 3 — Build Your Dataset (Week 3–4)

Goal: Capture and label images of trash items.

📘 Learn

Data collection best practices (consistent lighting, angles, background)

Folder structure for datasets

Data augmentation (flips, rotations, scaling)

🧩 Practice

Capture at least 200–300 images per class using your Pi or phone.
Classes:

dataset/
  ├─ paper/
  ├─ plastic/
  ├─ metal/
  └─ organic/


Use ImageDataGenerator to augment data.

Split into train/test (80%/20%).

🎓 Tools

OpenCV for capturing images

LabelImg (if you decide to extend to object detection later)

🧠 Stage 4 — Model Training (Week 4–5)

Goal: Train a CNN or use transfer learning with MobileNetV2.

📘 Learn

TensorFlow/Keras basics

Transfer learning and fine-tuning

Loss functions, accuracy metrics

🧩 Practice

Load a pretrained model:

base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))


Add classification layers for 4 classes.

Train and evaluate on your dataset.

Save as .h5 and convert to .tflite:

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("trash_classifier.tflite", "wb").write(tflite_model)

🎓 Resources

TensorFlow Transfer Learning Tutorial

Kaggle: Computer Vision Course

💻 Stage 5 — Deploy Model on Raspberry Pi (Week 6)

Goal: Run the .tflite model in real time on your Raspberry Pi.

📘 Learn

TensorFlow Lite Interpreter

Real-time inference with Pi Camera

Optimizing model performance (quantization, resizing input)

🧩 Practice

Install dependencies:

pip install tensorflow-lite opencv-python


Run inference:

import tensorflow as tf, cv2, numpy as np
interpreter = tf.lite.Interpreter(model_path="trash_classifier.tflite")
interpreter.allocate_tensors()


Show classification results on the live video feed with cv2.putText().

🎓 Resources

TensorFlow Lite Raspberry Pi Guide

YouTube: TensorFlow Lite on Raspberry Pi

🔌 Stage 6 — Hardware Integration (Week 7)

Goal: Control LEDs, servos, or motors based on detected class.

📘 Learn

Using Raspberry Pi GPIO pins with Python

Servo and relay control

Mapping AI outputs to hardware actions

🧩 Practice

Blink LEDs for each detected category.

Move servo to drop item into the correct bin.

if label == "Plastic": servo_pin.write(90)
elif label == "Paper": servo_pin.write(45)


Optional: use ultrasonic sensor to detect object presence.

🎓 Resources

GPIOZero Python Docs

Raspberry Pi Servo Motor Tutorial

⚡ Stage 7 — Optimization & Expansion (Week 8+)

Goal: Make it faster, more accurate, and smarter.

📘 Learn

Model quantization (INT8, FP16)

Using Google Coral TPU or Raspberry Pi 5 NPU

Combining non-vision sensors (metal detector, moisture sensor)

🧩 Practice

Quantize model with TensorFlow Lite converter.

Use hybrid approach:

Metal detector → quickly identify metallic waste

Camera AI → classify other types

🎓 Resources

TensorFlow Lite Optimization Guide

Coral USB Accelerator Docs

🚀 Stage 8 — Complete System Project

Goal: Build a working prototype.

🧩 Combine:

Raspberry Pi camera for classification

GPIO-controlled servos for sorting

Optional LCD/OLED display to show results

Enclosure with 4 bins (paper, plastic, metal, organic)

📘 Bonus Additions

Web dashboard using Flask or Streamlit

Data logging (how much of each category per day)

Add sound feedback (“Plastic detected!”)

🗓️ Suggested Timeline Summary
Week	Stage	Focus	Key Outcome
1–2	Foundations	Python + OpenCV + Pi setup	Capture and display images
2–3	ML Concepts	CNN + datasets	Understand image classification
3–4	Dataset	Collect 4-class dataset	Dataset ready for training
4–5	Model Training	MobileNetV2 fine-tuning	Trained .tflite model
6	Deployment	TensorFlow Lite on Pi	Real-time classification
7	Hardware Integration	GPIO + Servo + Sorting	Automated sorting
8+	Optimization	Quantization + Hybrid sensing	Faster, more reliable system
🧩 Optional Add-ons

Use YOLOv8 + Pi 5 for object detection.

Add cloud logging via Firebase.

Integrate Arduino for precise actuation control.
