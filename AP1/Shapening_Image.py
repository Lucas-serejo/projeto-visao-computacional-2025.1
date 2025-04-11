import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem original
imagem = cv2.imread("sua_imagem.jpg", cv2.IMREAD_GRAYSCALE)

# Aplicar um filtro Gaussiano para suavizar
suavizada = cv2.GaussianBlur(imagem, (9, 9), 0)

# Subtrair a imagem suavizada da original para realçar os detalhes
detalhes = cv2.subtract(imagem, suavizada)

# Aplicar um filtro Laplaciano para reforçar os detalhes
laplaciano = cv2.Laplacian(detalhes, cv2.CV_64F)

# Converter para escala de 8 bits
laplaciano = cv2.convertScaleAbs(laplaciano)

# Mostrar as imagens
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Imagem Original")
plt.imshow(imagem, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Detalhes Extraídos")
plt.imshow(detalhes, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Detalhes Reforçados")
plt.imshow(laplaciano, cmap="gray")

plt.show()

# Função para aplicar o filtro de Sharpening
def sharpening(imagem):
    # Aplicar um filtro Gaussiano para suavizar
    suavizada = cv2.GaussianBlur(imagem, (9, 9), 0)

    # Subtrair a imagem suavizada da original para realçar os detalhes
    detalhes = cv2.subtract(imagem, suavizada)

    # Aplicar um filtro Laplaciano para reforçar os detalhes
    laplaciano = cv2.Laplacian(detalhes, cv2.CV_64F)

    # Converter para escala de 8 bits
    laplaciano = cv2.convertScaleAbs(laplaciano)

    return laplaciano

# Função para mostrar os detalhes de uma imagem
def showDetails(imagem):
    # Aplicar um filtro Gaussiano para suavizar
    suavizada = cv2.GaussianBlur(imagem, (9, 9), 0)

    # Subtrair a imagem suavizada da original para realçar os detalhes
    detalhes = cv2.subtract(imagem, suavizada)

    return detalhes