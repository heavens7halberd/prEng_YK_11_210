import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, TimesformerForVideoClassification

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
model.eval()

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    frames = np.array(frames) / 255.0
    return frames

video_path = "typing.mp4"
frames = extract_frames(video_path)
inputs = processor(list(frames), return_tensors="pt", do_rescale=False)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
predicted_class_label = model.config.id2label[predicted_class_idx]
print("Прогнозируемый класс:", predicted_class_label)
