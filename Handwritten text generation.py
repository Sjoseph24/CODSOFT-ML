import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from keras import layers, models

# ==========================================
# 1. SETUP AND PATHS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "english.csv")
img_folder = current_dir 

print("⏳ Loading CSV data...")
df = pd.read_csv(csv_path)

# ==========================================
# 2. IMAGE PROCESSING
# ==========================================
print("⏳ Loading and resizing images (Optimized for 128x128)...")
images = df["image"]
output = []

for path in images:
    # Build the local path based on the CSV data
    full_image_path = os.path.join(img_folder, path)
    
    image = Image.open(full_image_path)
    image = image.convert("L")
    image = image.resize((128, 128))  # Resizing to 128x128 to save RAM
    image = np.array(image)
    output.append(image)

image_data = np.array(output)
image_data = image_data / 255.0  # Normalize pixel values
print(f"✅ Image data shape: {image_data.shape}")

# ==========================================
# 3. TEXT ENCODING
# ==========================================
print("⏳ Tokenizing labels...")
text_data = df["label"]

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts(text_data)

# Convert text data to sequences
text_sequences = tokenizer.texts_to_sequences(text_data)

# Padding sequences for a consistent input size
max_text_length = max(len(seq) for seq in text_sequences)
padded_text_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    text_sequences, maxlen=max_text_length
)

# ==========================================
# 4. BUILD AND TRAIN MODEL
# ==========================================
print("🏗️ Building the LSTM Image Generation Model...")
model = models.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50),
    layers.LSTM(1024),
    layers.Dense(128 * 128, activation='relu'),
    layers.Reshape((128, 128, 1))  # Output shape matched to 128x128
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("🧠 Training the model...")
# Note: Set to 10 epochs for testing. Increase to 300 for higher quality generation.
model.fit(np.array(padded_text_sequences), image_data, epochs=10, batch_size=64) 

# ==========================================
# 5. GENERATION FUNCTIONS
# ==========================================
def clear_and_create_dir(directory_path):
    """Safely clears an existing directory and recreates it."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path, ignore_errors=True)
    os.makedirs(directory_path, exist_ok=True)

def get_out_image(text="Sample", figsize=(10, 6), crop_size=(64, 128)):
    print(f"🎨 Generating image for text: '{text}'")
    textlist = list(text)
    predicted_images = []

    for new_text in textlist:
        if new_text == " ":
            white_image = np.ones((128, 128)) * 249  # White space character
            predicted_images.append(white_image)
        else:
            new_text_sequence = tokenizer.texts_to_sequences([new_text])
            padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
                new_text_sequence, maxlen=max_text_length
            )
            predicted_image = model.predict(padded_seq, verbose=0)[0]
            predicted_image = predicted_image.reshape(128, 128)
            predicted_images.append(predicted_image)

    out_dir = os.path.join(current_dir, 'out_images')
    clear_and_create_dir(out_dir)

    predicted_images = np.array(predicted_images)

    for new_text, image in zip(textlist, predicted_images):
        # Save individual character images
        image_path = os.path.join(out_dir, f"{new_text}_temp.png")
        fig = plt.figure(figsize=(5, 5))
        if new_text == " ":
            fig.figimage(image, vmin=0, vmax=255)
        else:
            fig.figimage(image)
        plt.savefig(image_path)
        plt.close(fig)

    def center_crop(image, target_size):
        width, height = image.size
        left = max(0, (width - target_size[0]) // 2)
        top = max(0, (height - target_size[1]) // 2)
        right = min(width, (width + target_size[0]) // 2)
        bottom = min(height, (height + target_size[1]) // 2)
        return image.crop((left, top, right, bottom))

    def merge_images(image_list):
        image_size = image_list[0].size
        merged_image = np.zeros((image_size[1], len(image_list) * image_size[0]))

        for i, img in enumerate(image_list):
            merged_image[:, i * image_size[0] : (i + 1) * image_size[0]] = np.array(img)

        return merged_image

    # Load generated images and center crop
    cropped_images = []
    for char in textlist:
        image_path = os.path.join(out_dir, f"{char}_temp.png")
        img = Image.open(image_path).convert("L") 
        cropped_img = center_crop(img, target_size=crop_size) 
        cropped_images.append(cropped_img)

    # Merge images into a single array
    merged_image_array = merge_images(cropped_images)

    # Display the final merged image
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(merged_image_array, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    plt.title(f"Generated: {text}")
    plt.show()

# ==========================================
# 6. RUN PREDICTION
# ==========================================
get_out_image("CodSoft", figsize=(6, 3))