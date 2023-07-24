import torch
from torchvision import transforms
import cv2
from PIL import Image
import mediapipe as mp

device = torch.device("cpu")  # Use the CPU

label_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

model = torch.load("signlang_model.pth", map_location=device)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def predict_image(image):
    image_tensor = data_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.item()
        predicted_label = label_dict[predicted_label]
        return predicted_label

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                min_x, min_y, max_x, max_y = frame.shape[1], frame.shape[0], 0, 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                if min_x < max_x and min_y < max_y:
                    hand_image = frame[min_y:max_y, min_x:max_x]

                    pil_image = Image.fromarray(hand_image)
                    predicted_label = predict_image(pil_image)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    color = (0, 255, 0)
                    text = f"Predicted Label: {predicted_label}"
                    cv2.putText(frame, text, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()