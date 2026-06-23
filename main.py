import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

OUTPUT_DIR = "output"  # diretório para salvar resultados
RANDOM_STATE = 42  # semente para reprodutibilidade


def load_and_convert(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"[ERRO] Imagem não encontrada: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return rgb, hsv


def extract_hue(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, :, 0].astype(np.float32) / 179.0  # normaliza para [0, 1] no OpenCV
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    hsv_normalized = np.dstack([h, s, v]).reshape(-1, 3)
    return hsv_normalized


def run_kmeans_elbow(
    hsv_features: np.ndarray, k_range: range, original_shape: tuple
) -> dict:
    results = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km.fit(hsv_features)
        labels_2d = km.labels_.reshape(original_shape[:2])
        results[k] = (km.inertia_, labels_2d)

    return results


def reconstruct_image_mean_rgb(
    rgb: np.ndarray, labels_2d: np.ndarray, k: int
) -> np.ndarray:
    pixels = rgb.reshape(-1, 3)
    labels_1d = labels_2d.flatten()

    new_pixels = np.zeros_like(pixels)

    for c in range(k):
        mask = labels_1d == c

        if np.any(mask):
            mean_color = pixels[mask].mean(axis=0)
            new_pixels[mask] = mean_color

    return new_pixels.reshape(rgb.shape)


def plot_elbow(k_range: range, results: dict, out_path: str):
    ks = list(k_range)
    inertias = [results[k][0] for k in ks]

    plt.figure(figsize=(8, 4))
    plt.plot(ks, inertias, marker="o", linewidth=2, color="steelblue")
    plt.title("Gráfico do Cotovelo — Baseado no Canal H (Hue)", fontsize=14)
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rgb, hsv = load_and_convert("input/imagem.jpg")

    hsv_features = extract_hue(hsv)

    k_range = range(2, 51)
    results = run_kmeans_elbow(hsv_features, k_range, rgb.shape)

    elbow_path = os.path.join(OUTPUT_DIR, "grafico.png")
    plot_elbow(k_range, results, elbow_path)

    for k, (_, labels_2d) in results.items():
        img_simplified = reconstruct_image_mean_rgb(rgb, labels_2d, k)
        fname = os.path.join(OUTPUT_DIR, f"kmeans_k{k:02d}.png")
        cv2.imwrite(fname, cv2.cvtColor(img_simplified, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
