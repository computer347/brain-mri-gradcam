import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 123
N_PER_CLASS = 200


class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
print(f"Classes: {class_names}")
 
def sample_paths(directory, n_per_class):
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(directory, cname)
        files = sorted(os.listdir(cdir))[:n_per_class]
        for f in files:
            paths.append(os.path.join(cdir, f))
            labels.append(idx)
    return paths, labels
 
def make_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE)
        return img, label
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    return ds
 
train_paths, train_labels = sample_paths(TRAIN_DIR, N_PER_CLASS)
test_paths, test_labels = sample_paths(TEST_DIR, N_PER_CLASS)
 
print(f"Training on {len(train_paths)} images, testing on {len(test_paths)} images")
 
train_ds_raw = make_dataset(train_paths, train_labels, shuffle=True)
test_ds_raw = make_dataset(test_paths, test_labels, shuffle=False)
 
normalization_layer = layers.Rescaling(1.0 / 255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))
 
input_shape = IMG_SIZE + (3,)
 
model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", name="last_conv"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(num_classes))
 
model.summary()
 
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
 
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
)
 
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
 
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(test_acc)
 
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy()), preds.numpy()[0]
 
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (img.shape[0], img.shape[1])).numpy().squeeze()
    cmap = plt.cm.jet(heatmap_resized)[..., :3]
    overlay = cmap * 255 * alpha + img * (1 - alpha)
    return overlay.astype("uint8")
 
sample_images, sample_labels = next(iter(test_ds_raw))
sample_images_norm = normalization_layer(sample_images)
 
n_show = min(9, sample_images.shape[0])
plt.figure(figsize=(12, 12))
for i in range(n_show):
    img_array = tf.expand_dims(sample_images_norm[i], axis=0)
    heatmap, pred_idx, logits = make_gradcam_heatmap(img_array, model, "last_conv")
    probs = tf.nn.softmax(logits).numpy()
    confidence = probs[pred_idx]
 
    original_img = sample_images[i].numpy().astype("uint8")
    overlay = overlay_heatmap(original_img, heatmap)
 
    plt.subplot(3, 3, i + 1)
    plt.imshow(overlay)
    true_label = class_names[sample_labels[i].numpy()]
    pred_label = class_names[pred_idx]
    color = "green" if pred_label == true_label else "red"
    plt.title(f"true: {true_label}\npred: {pred_label} ({confidence:.2f})", color=color, fontsize=9)
    plt.axis("off")
 
plt.tight_layout()
plt.show()
 