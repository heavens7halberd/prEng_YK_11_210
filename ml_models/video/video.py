import cv2
import numpy as np
import torch
import tempfile
from fastapi import UploadFile
from transformers import AutoImageProcessor, TimesformerForVideoClassification

def predict_video_class(video, num_frames=8):
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model.eval()

    if isinstance(video, UploadFile):
        video_bytes = video.file.read()
    else:
        video_bytes = video

    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
        temp_file.write(video_bytes)
        temp_file.flush()

        cap = cv2.VideoCapture(temp_file.name)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

        cap.release()

    frames_array = np.array(frames) / 255.0

    inputs = processor(list(frames_array), return_tensors="pt", do_rescale=False)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return model.config.id2label[logits.argmax(-1).item()]
