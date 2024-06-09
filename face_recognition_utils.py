import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
import os

# Инициализация моделей MTCNN и InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Загрузка известных лиц
known_face_encodings = []
known_face_names = []

path_to_images = 'st_images/'  # Убедитесь, что этот путь совпадает с вашей папкой изображений
student_images = os.listdir(path_to_images)

for image_name in student_images:
    image_path = os.path.join(path_to_images, image_name)
    img = cv2.imread(image_path)
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
    rgb_frame = frame[:, :, ::-1]  # Преобразуем цветовой формат BGR в RGB
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
            if distances[min_distance_index] < 0.6:  # Пороговое значение для распознавания
                names.append(known_face_names[min_distance_index])
            else:
                names.append("Unknown")

    return faces, names

def is_live(frame1, frame2, threshold=5.0):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > threshold:
            return True
    return False
 
