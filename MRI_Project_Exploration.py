import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
import numpy as np
from plots import save_png, jet, image_grid, training_curves_svg, confusion_matrix_svg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(os.path.join(RESULTS_DIR, "examples"), exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = int(os.environ.get("EPOCHS", 10))
SEED = 123
# Images per class; 0 = use every image in the folder
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 200))
EXAMPLES_PER_CLASS = 3

tf.keras.utils.set_random_seed(SEED)


class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)
print(f"Classes: {class_names}")

def sample_paths(directory, n_per_class):
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(directory, cname)
        files = sorted(os.listdir(cdir))
        if n_per_class:
            files = files[:n_per_class]
        for f in files:
            paths.append(os.path.join(cdir, f))
            labels.append(idx)
    return paths, labels

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, IMG_SIZE)

def make_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (load_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
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

training_curves_svg(os.path.join(RESULTS_DIR, "training_curves.svg"), history.history)

test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(test_acc)

# Confusion matrix (test_ds is not shuffled, so order matches test_labels)
y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
y_true = np.array(test_labels)
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
per_class_acc = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(class_names)}
print("Per-class accuracy:", per_class_acc)

confusion_matrix_svg(os.path.join(RESULTS_DIR, "confusion_matrix.svg"), cm, class_names,
                     f"Test accuracy: {test_acc:.1%}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Run the layers one by one so we can grab the last conv activation
    # (Keras 3 Sequential models have no symbolic .output to build a sub-model from)
    with tf.GradientTape() as tape:
        x = img_array
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
        preds = x
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
    cmap = jet(heatmap_resized)
    overlay = cmap * 255 * alpha + img * (1 - alpha)
    return overlay.astype("uint8")

# Grad-CAM on a few random test scans from every class
rng = np.random.default_rng(SEED)
example_idx = []
for c in range(num_classes):
    idx = np.where(y_true == c)[0]
    example_idx.extend(rng.choice(idx, EXAMPLES_PER_CLASS, replace=False))

examples = []
overlays = []
for i in example_idx:
    img = load_image(test_paths[i])
    img_array = tf.expand_dims(normalization_layer(img), axis=0)
    heatmap, pred_idx, logits = make_gradcam_heatmap(img_array, model, "last_conv")
    probs = tf.nn.softmax(logits).numpy()
    confidence = probs[pred_idx]

    original_img = img.numpy().astype("uint8")
    overlay = overlay_heatmap(original_img, heatmap)

    true_label = class_names[y_true[i]]
    pred_label = class_names[pred_idx]
    print(f"{os.path.basename(test_paths[i])}: true={true_label} pred={pred_label} ({confidence:.2f})")
    overlays.append(overlay)

    name = os.path.splitext(os.path.basename(test_paths[i]))[0]
    save_png(os.path.join(RESULTS_DIR, "examples", f"{name}_scan.png"), original_img)
    save_png(os.path.join(RESULTS_DIR, "examples", f"{name}_gradcam.png"), overlay)
    examples.append({
        "name": name,
        "true": true_label,
        "pred": pred_label,
        "confidence": float(confidence),
        "probs": {c: float(p) for c, p in zip(class_names, probs)},
    })

# One row per class, EXAMPLES_PER_CLASS columns
save_png(os.path.join(RESULTS_DIR, "gradcam_grid.png"), image_grid(overlays, EXAMPLES_PER_CLASS))

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump({
        "config": {"epochs": EPOCHS, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
                   "n_train": len(train_paths), "n_test": len(test_paths)},
        "classes": class_names,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "examples": examples,
    }, f, indent=2)
print(f"Saved results to {RESULTS_DIR}")
