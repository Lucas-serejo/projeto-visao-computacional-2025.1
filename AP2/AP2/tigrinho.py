import cv2
import time
from ultralytics import YOLO

# =============================
# CONFIGURAÇÕES
# =============================

model = YOLO(r'best.pt')  # seu modelo YOLOv8 treinado

card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

card_count = 0
last_detected_cards = {}  # {label: last_time_detected}
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
        card_number = label[:-1]    # Extrai '10' de '10H', 'A' de 'AD'

        if card_number not in card_values:
            continue

        # Obter o tempo da última contagem para esta carta específica (pelo label completo)
        last_time_this_card_was_counted = last_detected_cards.get(label, 0)

        # Desenhar caixa e texto para TODAS as detecções acima do threshold
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        value_for_display = card_values.get(card_number, "N/A")
        
        display_text = f'{label} [Valor: {value_for_display}] ({conf:.2f})'
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Lógica de contagem com o delay
        if (current_time - last_time_this_card_was_counted) < detection_delay:
            # Se foi contada recentemente, não a conte novamente, mas já foi desenhada.
            continue

        # Se passou pelo delay, então conte e atualize o tempo.
        card_count += card_values[card_number]
        last_detected_cards[label] = current_time # Usar 'label' para tratar cada carta (naipe e valor) unicamente no delay

    return frame

# =============================
# LOOP PRINCIPAL
# =============================

def main():
    cap = cv2.VideoCapture(0) # Ou o caminho para um vídeo

    if not cap.isOpened():
        print("Erro ao abrir a câmera ou vídeo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler o frame ou fim do vídeo.")
            break

        frame = detect_card_yolo(frame)

        # Escolher cor com base na contagem
        if card_count > 0:
            color = (0, 255, 0)  # verde
        elif card_count < 0:
            color = (0, 0, 255)  # vermelho
        else:
            color = (255, 255, 0)  # ciano/azul claro para neutro

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