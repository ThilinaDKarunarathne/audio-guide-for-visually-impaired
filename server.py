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

# Global variables to handle frame processing efficiently
latest_frame = None
processing_task = None

@app.route('/')
def serve_client():
    # Serve the client.html file from the static folder
    return send_from_directory(app.static_folder, 'client.html')

@app.route('/upload_frame', methods=['POST'])
async def upload_frame():
    global latest_frame, processing_task

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read image bytes and convert to numpy array
    img_bytes = request.files['image'].read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(image)

    # Update the latest frame
    latest_frame = frame

    # If there is an ongoing processing task, cancel it
    if processing_task and not processing_task.done():
        processing_task.cancel()

    # Start a new processing task
    processing_task = asyncio.create_task(process_frame())

    # Wait for the processing to complete
    try:
        result = await processing_task
        return jsonify(result)
    except asyncio.CancelledError:
        return jsonify({'status': 'Processing was canceled for a newer frame'}), 202

async def process_frame():
    global latest_frame

    # Process the latest frame
    current_frame = latest_frame
    latest_frame = None  # Mark as consumed
    command, processed_frame = await async_process_frame(current_frame)

    # Encode processed frame as base64
    _, buffer = cv2.imencode('.jpeg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'command': command,
        'processed_frame': processed_frame_base64
    }

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)