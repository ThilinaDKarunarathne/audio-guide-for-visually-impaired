from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import asyncio
from segformer_module import async_process_frame  # Ensure this file exists and has the async function

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_client():
    # Serve the client.html file from the static folder
    return send_from_directory(app.static_folder, 'client.html')

@app.route('/upload_frame', methods=['POST'])
async def upload_frame():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read image bytes and convert to numpy array
    img_bytes = request.files['image'].read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(image)

    # Call the asynchronous SegFormer logic and get command and processed frame
    command, processed_frame = await async_process_frame(frame)  # Updated to use async function

    # Encode processed frame as base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'command': command,
        'processed_frame': processed_frame_base64
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)