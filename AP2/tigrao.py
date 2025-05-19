import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# =============================
# CONFIGURAÇÕES E MODELO
# =============================

# Carrega o modelo treinado
model_path = '53tigres.h5'
model = load_model(model_path)

# Rótulos das cartas conforme a ordem de treino
labels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Valores das cartas para contagem
card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

# Variáveis de controle
card_count = 0
last_detected_card = None
last_detection_time = time.time()
detection_delay = 3  # segundos

# Região de interesse (ROI)
roi_x, roi_y, roi_width, roi_height = 200, 100, 300, 400

# =============================
# FUNÇÕES
# =============================

def detect_card_value_model(image):
    global card_count, last_detected_card, last_detection_time

    resized = cv2.resize(image, (200, 200))
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)

    predictions = model.predict(input_image)
    predicted_index = np.argmax(predictions)
    print(f"Predicted index: {predicted_index}, shape: {predictions.shape}")

    if predicted_index >= len(labels):
        print("Predicted index out of range!")
        return None

    predicted_label = labels[predicted_index]

    # Extraia apenas o número para contagem (ex: '10H' → '10')
    card_number = ''.join(filter(str.isdigit, predicted_label)) or predicted_label[0]

    current_time = time.time()
    if card_number == last_detected_card and (current_time - last_detection_time) < detection_delay:
        print(f"Ignoring duplicate card: {card_number}")
        return None

    card_count += card_values.get(card_number, 0)
    last_detected_card = card_number
    last_detection_time = current_time
    print(f"Detected card (model): {predicted_label}, Current count: {card_count}")

    return predicted_label



def localize_card(frame):
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if w * h > 5000 and 0.6 < aspect_ratio < 1.4:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            card_roi = roi[y:y + h, x:x + w]
            return card_roi, (x + roi_x, y + roi_y, w, h)

    return None, None


# =============================
# LOOP PRINCIPAL
# =============================

def main():
    global last_detection_time

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

        card_roi, bbox = localize_card(frame)

        if card_roi is not None:
            current_time = time.time()
            if (current_time - last_detection_time) >= detection_delay:
                card_value = detect_card_value_model(card_roi)
                if card_value is not None:
                    x, y, w, h = bbox
                    cv2.putText(frame, f"Card: {card_value}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"Card Count: {card_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Card Counting (Model)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# EXECUÇÃO
# =============================

if __name__ == "__main__":
    main()
