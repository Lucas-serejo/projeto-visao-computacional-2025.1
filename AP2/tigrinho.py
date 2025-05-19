import cv2
import time
from ultralytics import YOLO

# Inicializa o modelo treinado (substitua pelo caminho do seu .pt)
model = YOLO("baralho.pt")  # Ex: baralho.pt detecta cartas com classes "A", "2", ..., "K"

# Inicializa contagem
card_count = 0
card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

last_detected_card = None
last_detection_time = time.time()
detection_delay = 3  # segundos

def detect_card_yolo(image):
    global card_count, last_detected_card, last_detection_time

    results = model.predict(image, conf=0.5)
    current_time = time.time()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls].upper()

            if label in card_values:
                if label == last_detected_card and (current_time - last_detection_time) < detection_delay:
                    print(f"Ignorando duplicata: {label}")
                    continue

                card_count += card_values[label]
                last_detected_card = label
                last_detection_time = current_time
                print(f"Detectado: {label}, Contagem atual: {card_count}")
                return label, box.xyxy[0].tolist()

    return None, None

def main():
    global last_detection_time

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        card_label, bbox = detect_card_yolo(frame)

        if card_label is not None and bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{card_label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostrar contagem
        cv2.putText(frame, f"Card Count: {card_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Card Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
