import cv2
import time
from ultralytics import YOLO

# =============================
# CONFIGURAÇÕES
# =============================

model = YOLO('caminho/para/o modelo')  # seu modelo YOLOv8 treinado

card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

card_count = 0
last_detected_cards = {}  # {card_number: last_time_detected}
detection_delay = 3  # segundos
CONFIDENCE_THRESHOLD = 0.5

# =============================
# FUNÇÃO DE DETECÇÃO
# =============================

def detect_card_yolo(frame):
    global card_count, last_detected_cards

    results = model(frame)[0]
    current_time = time.time()

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]  # Ex: '10H', 'AD'
        card_number = label[:-1]

        if card_number not in card_values:
            continue

        # Prevenir múltiplas contagens da mesma carta em sequência
        last_time = last_detected_cards.get(card_number, 0)
        if (current_time - last_time) < detection_delay:
            continue

        card_count += card_values[card_number]
        last_detected_cards[card_number] = current_time

        # Desenhar caixa e texto
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # (Opcional) Salvar imagem da carta detectada
        # card_crop = frame[y1:y2, x1:x2]
        # cv2.imwrite(f'detected_{label}_{int(current_time)}.jpg', card_crop)

    return frame

# =============================
# LOOP PRINCIPAL
# =============================

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_card_yolo(frame)

        # Escolher cor com base na contagem
        if card_count > 0:
            color = (0, 255, 0)  # verde
        elif card_count < 0:
            color = (0, 0, 255)  # vermelho
        else:
            color = (255, 255, 0)  # azul

        cv2.putText(frame, f"Card Count: {card_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Card Detection - YOLOv8", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# EXECUÇÃO
# =============================

if __name__ == "__main__":
    main()
