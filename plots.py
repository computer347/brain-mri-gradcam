"""Dependency-free output helpers: PNG via TensorFlow, charts as plain SVG."""
import numpy as np
import tensorflow as tf


def save_png(path, img):
    tf.io.write_file(path, tf.io.encode_png(np.asarray(img, dtype=np.uint8)))


def jet(x):
    """Approximate matplotlib's 'jet' colormap for values in [0, 1] -> RGB in [0, 1]."""
    x = np.clip(x, 0, 1)[..., None]
    return np.clip(1.5 - np.abs(4 * x - np.array([3, 2, 1])), 0, 1)


def image_grid(images, n_cols, gap=6, bg=255):
    h, w, _ = images[0].shape
    n_rows = int(np.ceil(len(images) / n_cols))
    grid = np.full((n_rows * h + (n_rows - 1) * gap, n_cols * w + (n_cols - 1) * gap, 3), bg, np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, n_cols)
        grid[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = img
    return grid


FONT = 'font-family="Helvetica, Arial, sans-serif"'
COLORS = {"train": "#2a6fdb", "test": "#e8711c"}


def _line_panel(x0, title, series, y_max, w=360, h=240):
    pad_l, pad_b, pad_t = 44, 30, 26
    pw, ph = w - pad_l - 12, h - pad_b - pad_t
    n = len(next(iter(series.values())))
    sx = lambda i: x0 + pad_l + (pw * i / max(n - 1, 1))
    sy = lambda v: pad_t + ph - ph * v / y_max
    out = [f'<text x="{x0 + w / 2}" y="16" text-anchor="middle" font-size="14" font-weight="600" {FONT}>{title}</text>']
    for k in range(5):
        v = y_max * k / 4
        out.append(f'<line x1="{x0 + pad_l}" x2="{x0 + pad_l + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{x0 + pad_l - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#555" {FONT}>{v:.2g}</text>')
    for i in range(n):
        out.append(f'<text x="{sx(i):.1f}" y="{pad_t + ph + 16}" text-anchor="middle" font-size="11" fill="#555" {FONT}>{i + 1}</text>')
    for j, (name, vals) in enumerate(series.items()):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[name]}" stroke-width="2.5"/>')
        ly = pad_t + 8 + 16 * j
        out.append(f'<line x1="{x0 + pad_l + pw - 70}" x2="{x0 + pad_l + pw - 52}" y1="{ly}" y2="{ly}" stroke="{COLORS[name]}" stroke-width="2.5"/>')
        out.append(f'<text x="{x0 + pad_l + pw - 46}" y="{ly + 4}" font-size="11" {FONT}>{name}</text>')
    out.append(f'<text x="{x0 + pad_l + pw / 2}" y="{h - 2}" text-anchor="middle" font-size="11" fill="#555" {FONT}>epoch</text>')
    return out


def training_curves_svg(path, history):
    loss_max = max(max(history["loss"]), max(history["val_loss"])) * 1.1
    parts = _line_panel(0, "Accuracy", {"train": history["accuracy"], "test": history["val_accuracy"]}, 1.0)
    parts += _line_panel(380, "Loss", {"train": history["loss"], "test": history["val_loss"]}, loss_max)
    _write_svg(path, 740, 240, parts)


def confusion_matrix_svg(path, cm, class_names, title):
    n, cell, left, top = len(class_names), 80, 110, 50
    w, h = left + n * cell + 10, top + n * cell + 70
    parts = [f'<text x="{left + n * cell / 2}" y="22" text-anchor="middle" font-size="15" font-weight="600" {FONT}>{title}</text>']
    vmax = cm.max() or 1
    for i in range(n):
        for j in range(n):
            t = cm[i, j] / vmax
            r, g, b = (int(247 - t * (247 - 8)), int(251 - t * (251 - 48)), int(255 - t * (255 - 107)))
            x, y = left + j * cell, top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({r},{g},{b})" stroke="#fff"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 6}" text-anchor="middle" font-size="17" '
                         f'fill="{"#fff" if t > 0.5 else "#111"}" {FONT}>{cm[i, j]}</text>')
        parts.append(f'<text x="{left - 8}" y="{top + i * cell + cell / 2 + 4}" text-anchor="end" font-size="12" {FONT}>{class_names[i]}</text>')
        parts.append(f'<text x="{left + i * cell + cell / 2}" y="{top + n * cell + 18}" text-anchor="middle" font-size="12" {FONT}>{class_names[i]}</text>')
    parts.append(f'<text x="{left + n * cell / 2}" y="{h - 14}" text-anchor="middle" font-size="12" fill="#555" {FONT}>predicted</text>')
    parts.append(f'<text x="14" y="{top + n * cell / 2}" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90 14 {top + n * cell / 2})" {FONT}>true</text>')
    _write_svg(path, w, h, parts)


def _write_svg(path, w, h, parts):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
                f'<rect width="100%" height="100%" fill="#fff"/>' + "".join(parts) + "</svg>")
