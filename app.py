from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
import threading
import time
import base64
import string

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5501"],  # Your frontend URL
        "allow_headers": ["Content-Type"],
        "methods": ["POST", "GET", "OPTIONS"]
    }
})

class HandGestureRecognizer:
    def __init__(self):
        # Load models and labels
        with open('./classes/ALPH_CLASSES.json', 'r') as json_file:
            self.alph_labels = json.load(json_file)
        self.alph_labels = {int(k): v for k, v in self.alph_labels.items()}

        with open('./classes/NUM_CLASSES.json', 'r') as json_file:
            self.num_labels = json.load(json_file)
        self.num_labels = {int(k): v for k, v in self.num_labels.items()}

        self.alph_model = pickle.load(open('./models/alph_model.p', 'rb'))['model']
        self.num_model = pickle.load(open('./models/num_model.p', 'rb'))['model']

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
        
        # Game state variables
        self.current_model = None
        self.current_labels = None
        self.current_model_type = None
        self.current_targets = []
        self.current_target_index = 0
        self.last_correct_prediction_time = 0
        self.prediction_lock = threading.Lock()

    def load_model(self, model_type):
        with self.prediction_lock:
            if model_type == 'alphabet':
                self.current_model = self.alph_model
                self.current_labels = self.alph_labels
                self.current_model_type = 'alphabet'
                self.current_targets = list(string.ascii_uppercase)
            elif model_type == 'number':
                self.current_model = self.num_model
                self.current_labels = self.num_labels
                self.current_model_type = 'number'
                self.current_targets = list(map(str, range(0, 11)))
            
            self.current_target_index = 0
            return self.current_targets[0]

    def predict_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks and self.current_model is not None:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            data_aux = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            prediction = self.current_model.predict([np.asarray(data_aux)])
            predicted_character = self.current_labels[int(prediction[0])]
            
            return predicted_character, hand_landmarks
        
        return None, None

    def check_prediction(self, predicted_character):
        with self.prediction_lock:
            current_time = time.time()
            if str(predicted_character) == str(self.current_targets[self.current_target_index]):
                if current_time - self.last_correct_prediction_time > 1:
                    self.last_correct_prediction_time = current_time
                    self.current_target_index += 1
                    
                    if self.current_target_index >= len(self.current_targets):
                        return True, None
                    
                    return True, self.current_targets[self.current_target_index]
        return False, None

# Global recognizer instance
recognizer = HandGestureRecognizer()

@app.route('/')
def index():
    """Testing interface"""
    return render_template('index.html')

@app.route('/embed')
def embed():
    """Camera embed endpoint"""
    return render_template('camera_embed.html')

@app.route('/api/load_model/<model_type>')
def load_model(model_type):
    target = recognizer.load_model(model_type)
    response = jsonify(success=True, target=target)
    response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5501')
    return response

@app.route('/api/predict', methods=['POST'])
def predict():
    frame_data = request.json.get('frame')
    try:
        nparr = np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        predicted_character, hand_landmarks = recognizer.predict_gesture(frame)
        
        if predicted_character:
            is_correct, new_target = recognizer.check_prediction(predicted_character)
            return jsonify({
                'predicted_character': predicted_character,
                'is_correct': is_correct,
                'new_target': new_target
            })
        
        return jsonify({'predicted_character': None})
    except Exception as e:
        print(f"Error in predict: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)