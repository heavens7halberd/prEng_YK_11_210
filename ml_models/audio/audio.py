from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch
import io

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(file):
    try:
        audio, sample_rate = librosa.load(io.BytesIO(file), sr=16000)

        input_values = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_text = processor.batch_decode(predicted_ids)[0]

        return decoded_text
    except Exception as e:
        return {"error": str(e)}
