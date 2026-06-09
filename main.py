import cv2
import numpy as np
from sklearn.cluster import KMeans  # <-- Importação do K-Means


def prepare_image(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found.")
        return None

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        dst = np.empty_like(image)
        cv2.normalize(image, dst, 0, 255, cv2.NORM_MINMAX)
        image = dst.astype(np.uint8)

    return image


def rgb_to_hsi(image):
    with np.errstate(divide="ignore", invalid="ignore"):
        img_norm = image.astype(np.float32) / 255.0
        R, G, B = cv2.split(img_norm)
        I = (R + G + B) / 3.0

        min_rgb = np.minimum(np.minimum(R, G), B)
        S = 1.0 - (3.0 / (R + G + B + 1e-6)) * min_rgb
        S[I == 0] = 0

        numerador = 0.5 * ((R - G) + (R - B))
        denominador = np.sqrt((R - G) ** 2 + (R - B) * (G - B))

        theta = np.arccos(numerador / (denominador + 1e-6))
        H = np.degrees(theta)

        H[B > G] = 360.0 - H[B > G]
        H[denominador == 0] = 0
        H_8bit = (H / 2).astype(np.uint8)

        S_8bit = (S * 255).astype(np.uint8)
        I_8bit = (I * 255).astype(np.uint8)

        img_hsi = cv2.merge([H_8bit, S_8bit, I_8bit])
    return img_hsi


def apply_kmeans(imagem, k=8):
    # 1. Salvar as dimensões originais (Altura, Largura, Canais)
    h, w, c = imagem.shape

    # 2. Achatar a imagem 3D em uma matriz 2D (Lista de pixels, Canais)
    pixels = imagem.reshape((-1, c))

    # 3. Inicializar o K-Means e ajustar (treinar) com os pixels
    # n_init="auto" evita avisos de versões mais recentes do sklearn
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(pixels)

    # 4. Obter as cores centrais de cada grupo e convertê-las para uint8 (0-255)
    centros = kmeans.cluster_centers_.astype(np.uint8)

    # 5. Pegar as marcações (rótulos) do K-Means e mapeá-las para os centros
    rotulos = kmeans.labels_
    imagem_agrupada = centros[rotulos]

    # 6. Reconstruir a imagem para o seu formato 3D original
    imagem_agrupada = imagem_agrupada.reshape((h, w, c))

    return imagem_agrupada


def main():
    imagePath = "./data/paisagem.png"

    image_8bit = prepare_image(imagePath)
    if image_8bit is None:
        return

    imagem_kmeans = apply_kmeans(image_8bit, k=15)

    imagem_kmeans_bgr = cv2.cvtColor(imagem_kmeans, cv2.COLOR_RGB2BGR)
    cv2.imwrite("resultado_kmeans_rgb1.png", imagem_kmeans_bgr)

    image_hsi = rgb_to_hsi(image_8bit)

    img_para_exibir = image_hsi

    # cv2.imwrite("resultado_hsi_completo.png", img_para_exibir)

    # Separar os canais
    H, S, I = cv2.split(img_para_exibir)

    # Salvar os canais individualmente
    # cv2.imwrite("resultado_1_matiz.png", H)
    # cv2.imwrite("resultado_2_saturacao.png", S)
    # cv2.imwrite("resultado_3_intensidade.png", I)


if __name__ == "__main__":
    main()
