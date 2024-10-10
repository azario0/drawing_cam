import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set up the drawing canvas
canvas = None
drawing = False
last_x, last_y = 0, 0

# Define colors and initialize current_color as None
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
current_color = None

# Create color palette
color_palette = np.zeros((50, 300, 3), dtype=np.uint8)
for i, color in enumerate(colors):
    color_palette[:, i*50:(i+1)*50] = color

def is_in_color_palette(x, y):
    return 0 <= x < 300 and 0 <= y < 50

def is_in_reset_button(x, y):
    return 300 <= x < 350 and 0 <= y < 50

def draw_ui(frame):
    frame[:50, :300] = color_palette
    cv2.rectangle(frame, (300, 0), (350, 50), (128, 128, 128), -1)
    cv2.putText(frame, "Reset", (305, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if current_color is None:
        cv2.putText(frame, "Select a color to start drawing", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def process_hand(hand_landmarks, frame_shape):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_finger_tip.x * frame_shape[1]), int(index_finger_tip.y * frame_shape[0])
    
    fingers_raised = sum([hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y for i in [8, 12, 16, 20]])
    
    return x, y, fingers_raised

def main():
    global canvas, drawing, last_x, last_y, current_color

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        
        if canvas is None:
            canvas = np.zeros(frame.shape, dtype=np.uint8)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        draw_ui(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x, y, fingers_raised = process_hand(hand_landmarks, frame.shape)
                
                if fingers_raised >= 4:  # Erase
                    cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                elif is_in_color_palette(x, y):  # Change color
                    current_color = tuple(map(int, frame[y, x]))
                elif is_in_reset_button(x, y):  # Reset canvas
                    canvas.fill(0)
                    current_color = None
                elif current_color:  # Draw only if a color is selected
                    if drawing:
                        cv2.line(canvas, (last_x, last_y), (x, y), current_color, 5)
                    drawing = True
                    last_x, last_y = x, y
                
                cv2.circle(frame, (x, y), 10, current_color or (200, 200, 200), -1)
            
        else:
            drawing = False
        
        result = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
        cv2.imshow("Finger Drawing", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()