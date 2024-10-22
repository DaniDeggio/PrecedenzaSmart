import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import socket
from flask import Flask, Response
from collections import defaultdict
from threading import Thread

# Thread for video capture to improve performance by separating video acquisition from processing
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Configurazione socket TCP
TCP_IP = '127.0.0.1'  # IP del server
TCP_PORT = 5005       # Porta del server
BUFFER_SIZE = 1024    # Dimensione buffer

# Creazione del socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

# Creazione app Flask
app = Flask(__name__)

# Modello YOLO
model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite', task='detect')

# Caricamento classi COCO
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Funzione per elaborare i frame separata dal server
def process_video():
    frame_count = 0
    start_time = time.time()
    total_inference_time = 0
    inference_frame_count = 0

    while True:
        ret, frame = vs.read()  # Usa lo stream video
        if not ret:
            break

        frame_count += 1
        if frame_count % 6 != 0:
            continue

        resized_frame = cv2.resize(frame, (320, 240))

        inference_start = time.time()
        results = model.predict(resized_frame, imgsz=240)
        inference_end = time.time()

        total_inference_time += (inference_end - inference_start)
        inference_frame_count += 1

        label_count = defaultdict(int)

        if len(results) > 0:
            a = results[0].boxes.data
            if a is not None and len(a) > 0:
                px = pd.DataFrame(a).astype("float")

                for index, row in px.iterrows():
                    x1 = int(row[0] * (frame.shape[1] / 320))
                    y1 = int(row[1] * (frame.shape[0] / 240))
                    x2 = int(row[2] * (frame.shape[1] / 320))
                    y2 = int(row[3] * (frame.shape[0] / 240))
                    d = int(row[5])
                    c = class_list[d]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                    # Incrementa il conteggio
                    label_count[c] += 1

                    # Invia il nome dell'oggetto riconosciuto tramite TCP
                    message = f"{d},{c}\n"
                    s.send(message.encode('utf-8'))

        # Calcolo FPS e visualizzazione sullo schermo
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        cvzone.putTextRect(frame, f'FPS (visual): {round(fps, 2)}', (10, 30), 1, 1)

        if inference_frame_count > 0:
            inference_fps = inference_frame_count / total_inference_time
            cvzone.putTextRect(frame, f'FPS (inference): {round(inference_fps, 2)}', (10, 60), 1, 1)

# Funzione per generare lo stream video per il server Flask
def generate_frames():
    while True:
        ret, frame = vs.read()  # Legge dal thread del video
        if not ret:
            break

        # Codifica il frame come JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream del frame via HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

# Avvia lo stream video in un thread separato
vs = VideoStream().start()

# Avvia il thread per elaborare i frame video
processing_thread = Thread(target=process_video)
processing_thread.start()

# Avvia il server Flask in un terzo thread
flask_thread = Thread(target=run_flask)
flask_thread.start()

# Ora ci sono tre thread in esecuzione: uno per la cattura del video, uno per elaborarlo e uno per il server Flask.
