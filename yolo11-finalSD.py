import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import socket
from flask import Flask, Response
import threading

# Configurazione della fotocamera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Caricamento dei nomi delle classi COCO
with open("coco.txt", "r") as f:
        class_names = f.read().splitlines()

# Caricamento del modello YOLOv8
model = YOLO("yolo11n.pt")

# Configurazione del socket TCP
TCP_IP = '127.0.0.1'  # Indirizzo IP del server
TCP_PORT = 5005       # Porta del server
BUFFER_SIZE = 1024    # Dimensione del buffer

# Creazione del socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

# Creazione dell'app Flask
app = Flask(__name__)

def generate_frames():
        count = 0
        while True:
                frame = picam2.capture_array()

                count += 1
                if count % 2 != 0:
                        continue
                frame = cv2.flip(frame, -1)

                # Esecuzione del tracking YOLOv8 sul frame
                results = model.track(frame, persist=True, imgsz=256)

                # Configurazione del socket TCP
                TCP_IP = '127.0.0.1'  # Indirizzo IP del server
                TCP_PORT = 5005       # Porta del server
                BUFFER_SIZE = 1024    # Dimensione del buffer

                # Creazione del socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((TCP_IP, TCP_PORT))

                # Controllo se ci sono box nei risultati
                if results[0].boxes is not None and results[0].boxes.id is not None:
                        # Ottenimento dei box (x, y, w, h), ID delle classi, ID dei track e confidenze
                        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
                        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
                        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
                        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

                        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                                c = class_names[class_id]
                                x1, y1, x2, y2 = box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                                # Invia il nome dell'oggetto riconosciuto tramite TCP
                                message = f"{track_id},{c}\n"
                                s.send(message.encode('utf-8'))

                # Chiudi il socket TCP
                s.close()

                # Codifica il frame come JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Genera il frame come stream
                yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
        app.run(host='0.0.0.0', port=5000)

# Esegui il server Flask in un thread separato
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()