import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
import os

# Initialize MTCNN and InceptionResnetV1 models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load known faces
known_face_encodings = []
known_face_names = []

path_to_images = 'st_images/'  # Ensure this path matches your images folder
student_images = os.listdir(path_to_images)

for image_name in student_images:
    image_path = os.path.join(path_to_images, image_name)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces, _ = mtcnn.detect(img_rgb)
    
    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = face
            face_img = img_rgb[int(y1):int(y2), int(x1):int(x2)]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            encoding = resnet(face_img).detach().numpy().flatten()
            known_face_encodings.append(encoding)
            known_face_names.append(image_name.split('.')[0])

def recognize_face(frame):
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    faces, _ = mtcnn.detect(rgb_frame)
    names = []

    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = face
            face_img = rgb_frame[int(y1):int(y2), int(x1):int(x2)]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            encoding = resnet(face_img).detach().numpy().flatten()
            
            distances = np.linalg.norm(known_face_encodings - encoding, axis=1)
            min_distance_index = np.argmin(distances)
            if distances[min_distance_index] < 0.6:  # Threshold for recognition
                names.append(known_face_names[min_distance_index])
            else:
                names.append("Unknown")

    return faces, names

def is_live(frame1, frame2, threshold=5.0):
    # Check for frame difference
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) > threshold:
                return True
    
    return False

def detect_blinks(frame, eyes):
    # Placeholder function for blink detection
    # Here, you should add your blink detection algorithm, e.g., based on eye aspect ratio
    pass

def main():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        faces, names = recognize_face(frame2)
        if is_live(frame1, frame2):
            for (face, name) in zip(faces, names):
                x1, y1, x2, y2 = face
                cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame2, name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            cv2.putText(frame2, "This is not a live person!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Video', frame2)
        frame1 = frame2.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
