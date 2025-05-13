import cv2
import numpy as np
import onnxruntime as ort
from transformers import SegformerImageProcessor
import asyncio

# Load the ONNX model
onnx_model_path = "segformer_b0_finetuned_cityscapes_768_768.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Load the feature extractor
model_name = "nvidia/segformer-b0-finetuned-cityscapes-768-768"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)

# Colors and classes
COLOR_MAP = {
    0: (255, 0, 0),    # Road
    1: (150, 150, 150), # Sidewalk
    5: (0, 255, 255),   # Pole
    11: (0, 255, 0),    # Person
    13: (0, 0, 255),    # Vehicle
    14: (0, 0, 255),    # Vehicle
    15: (0, 0, 255),    # Vehicle
    17: (0, 0, 255),    # Vehicle
    18: (0, 0, 255)     # Vehicle
}

CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    5: "pole",
    11: "person",
    13: "vehicle",
    14: "vehicle",
    15: "vehicle",
    17: "vehicle",
    18: "vehicle"
}

async def async_process_frame(frame):
    # Resize and preprocess the frame
    frame = cv2.resize(frame, (768, 768))
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess using the feature extractor
    inputs = feature_extractor(images=input_image, return_tensors="np")
    input_array = inputs["pixel_values"]

    # Run inference asynchronously
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    logits = await asyncio.to_thread(ort_session.run, None, ort_inputs)
    logits = logits[0]  # Get logits

    # Post-process logits
    raw_predicted = np.argmax(logits, axis=1)[0]

    # Resize prediction to original frame size
    predicted = cv2.resize(raw_predicted.astype(np.uint8), (768, 768), interpolation=cv2.INTER_NEAREST)

    # Create a blended overlay
    overlay = np.zeros_like(frame)
    for class_id, color in COLOR_MAP.items():
        overlay[predicted == class_id] = color
    blended_frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    # Grid analysis
    h, w = frame.shape[:2]
    rows, cols = 5, 5
    box_h, box_w = h // rows, w // cols
    grid_analysis = []

    for i in range(rows):
        row = []
        for j in range(cols):
            y1, y2 = i * box_h, (i + 1) * box_h
            x1, x2 = j * box_w, (j + 1) * box_w
            region = predicted[y1:y2, x1:x2]
            hist = {class_id: np.sum(region == class_id) for class_id in COLOR_MAP.keys()}
            total = region.size
            row.append({k: v / total for k, v in hist.items()})
        grid_analysis.append(row)

    # Decision-making logic
    center_box = grid_analysis[4][2]
    left_boxes = [grid_analysis[4][0], grid_analysis[4][1]]
    right_boxes = [grid_analysis[4][3], grid_analysis[4][4]]

    center_road = center_box.get(0, 0)
    center_sidewalk = center_box.get(1, 0)
    center_person = center_box.get(11, 0)
    center_pole = center_box.get(5, 0)

    # New: Sum all vehicle probabilities
    vehicle_classes = [13, 14, 15, 17, 18]
    center_vehicle = sum(center_box.get(cls, 0) for cls in vehicle_classes)

    left_road = any(box.get(0, 0) > 0.5 for box in left_boxes)
    right_road = any(box.get(0, 0) > 0.5 for box in right_boxes)

    left_sidewalk = any(box.get(1, 0) > 0.5 for box in left_boxes)
    right_sidewalk = any(box.get(1, 0) > 0.5 for box in right_boxes)

    # Command logic
    if center_road >= 0.9:
        command = "Go straight. You're on the road."
    elif center_road < 0.9 and left_road:
        command = "Not on the road. Turn left."
    elif center_road < 0.9 and right_road:
        command = "Not on the road. Turn right."
    elif center_sidewalk >= 0.9:
        command = "Go straight. You're on the sidewalk."
    elif center_sidewalk < 0.9 and left_sidewalk:
        command = "Not on sidewalk. Turn left to sidewalk."
    elif center_sidewalk < 0.9 and right_sidewalk:
        command = "Not on sidewalk. Turn right to sidewalk."
    elif center_person > 0.9:
        if left_sidewalk:
            command = "Person ahead. Turn left to sidewalk."
        elif right_sidewalk:
            command = "Person ahead. Turn right to sidewalk."
        elif left_road:
            command = "Person ahead. Turn left on the road."
        elif right_road:
            command = "Person ahead. Turn right on the road."
        else:
            command = "Person ahead. Stop."
    elif center_vehicle >= 0.9:
        command = "Vehicle ahead. Stop."
    elif center_pole > 0.4:
        if left_road:
            command = "Pole in front. Turn left."
        elif right_road:
            command = "Pole in front. Turn right."
        else:
            command = "Pole in front. Stop."
    elif not left_sidewalk and not right_sidewalk and not left_road and not right_road:
        most_common = max(center_box.items(), key=lambda x: x[1])[0]
        if most_common == 5:
            command = "Pole in front. Stop."
        elif most_common == 11:
            command = "Person ahead. Stop."
        elif most_common in [0, 1]:
            command = "Stop. Cannot find a road or sidewalk."
        else:
            command = f"Stop. There is a {CLASS_NAMES.get(most_common, 'barrier')} in front."
    else:
        most_common = max(center_box.items(), key=lambda x: x[1])[0]
        command = f"Stop. There is a {CLASS_NAMES.get(most_common, 'barrier')} in front."

    return command, blended_frame
