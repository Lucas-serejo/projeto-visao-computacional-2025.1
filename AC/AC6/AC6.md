# Comparativo de Detecção de Objetos: MobileNet SSD vs YOLOv8

Este experimento compara os resultados de dois modelos de detecção de objetos aplicados em imagens com objetos diversos, como eletrodomésticos e utensílios.

---

## MobileNet SSD

### Configurações:
- **Modelo**: SSD MobileNet v2 (COCO)
- **Framework**: OpenCV DNN
- **Resolução da entrada**: 300x300
- **Confidence threshold**: `0.4`
- **Imagem**: `cozinha.jpg`

### Tempo de Execução
- **Tempo total estimado**: **28.07 ms**

### Objetos Detectados

| Classe COCO detectada | Confiança |
|-----------------------|-----------|
| Oven                 | 0.98      |
| Oven (2ª instância)  | 0.83      |
| Toaster              | 0.78      |
| Bed                  | 0.75      |


---

## YOLOv8

### Configurações:
- **Modelo**: YOLOv8 (Ultralytics)
- **Framework**: Ultralytics API (PyTorch)
- **Tamanho da entrada**: 640x640
- **Confidence threshold**: `0.25` (default)
- **Imagem**: `cozinha.jpg`

### Tempo de Execução

| Etapa              | Tempo   |
|-------------------|---------|
| Pré-processamento  | 2.9 ms  |
| Inferência         | 44.8 ms |
| Pós-processamento  | 0.8 ms  |
| **Total**          | **~48.5 ms** |

### Objetos Detectados

| Classe YOLOv8      | Quantidade |
|--------------------|------------|
| Bottle             | 2          |
| Cup                | 1          |
| Potted Plant       | 1          |
| Microwave          | 2          |
| Oven               | 2          |

---

## Conclusão Comparativa

| Critério                   | MobileNet SSD           | YOLOv8                      |
|----------------------------|--------------------------|-----------------------------|
| Facilidade de uso          | Alta (OpenCV)            | Alta (Ultralytics)          |
| Precisão em objetos reais  | Boa, mas limitada        | Alta                        |
| Velocidade                 | **28.07 ms**             | **~48.5 ms**                |
| Configurável               | Médio                    | Alto (`conf`, `iou`, etc.)  |
| Robustez                   | Limitado ao COCO         | Adaptável a múltiplos domínios |

**YOLOv8 se destaca em versatilidade e precisão**, sendo ideal para aplicações modernas.   Já o **MobileNet SSD** entrega bons resultados com maior velocidade, útil para dispositivos com menos recursos.

---
