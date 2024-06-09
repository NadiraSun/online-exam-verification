from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from face_recognition_utils import recognize_face, is_live

app = Flask(__name__)

camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html', identified=None)

@app.route('/identify', methods=['POST'])
def identify():
    ret, frame = camera.read()
    if not ret:
        return render_template('index.html', identified=False)
    
    # Уменьшаем разрешение кадра для ускорения обработки
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    face_locations, face_names = recognize_face(small_frame)
    
    if face_names and "Unknown" not in face_names:
        return render_template('index.html', identified=True)
    else:
        return render_template('index.html', identified=False)

@app.route('/start_exam', methods=['POST'])
def start_exam():
    return render_template('exam_started.html')

def gen_frames():
    ret, prev_frame = camera.read()
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Уменьшаем разрешение кадра для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        face_locations, face_names = recognize_face(small_frame)
        
        if face_locations is not None:
            for (x1, y1, x2, y2), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (int(x1*2), int(y1*2)), (int(x2*2), int(y2*2)), (0, 0, 255), 2)
                # Изменен цвет текста на черный
                cv2.putText(frame, name, (int(x1*2) + 6, int(y2*2) - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
        
        if is_live(prev_frame, frame):
            cv2.putText(frame, "Live", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Live", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        prev_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
 
