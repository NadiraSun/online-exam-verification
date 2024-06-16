from flask import Flask, render_template, Response, request
import cv2
from face_recognition_utils import recognize_face, is_live, load_known_faces

app = Flask(__name__)

camera = cv2.VideoCapture(0)
load_known_faces('st_images')

@app.route('/')
def index():
    return render_template('index.html', identified=None)

@app.route('/identify', methods=['POST'])
def identify():
    ret, frame = camera.read()
    if not ret:
        print("Ошибка: не удается захватить кадр")
        return render_template('index.html', identified=False)

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    face_locations, face_names, _ = recognize_face(small_frame)

    if face_names and len(face_names) == 1 and "Unknown" not in face_names:
        return render_template('index.html', identified=True)
    else:
        return render_template('index.html', identified=False)

@app.route('/start_exam', methods=['POST'])
def start_exam():
    return render_template('exam_started.html')

def gen_frames():
    ret, prev_frame = camera.read()
    if not ret:
        print("Ошибка: не удается захватить начальный кадр")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Ошибка: не удается захватить кадр")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face_locations, face_names, similarities = recognize_face(small_frame)

        if face_locations is not None and len(face_locations) == 1:
            (x1, y1, x2, y2), name, similarity = face_locations[0], face_names[0], similarities[0]
            x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{name} ({similarity * 100:.2f}%)"
            cv2.putText(frame, text, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)

        if is_live(prev_frame, frame):
            cv2.putText(frame, "Live", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Live", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        prev_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Ошибка: не удается кодировать кадр")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        if camera.isOpened():
            camera.release()
