from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000)
        print(f"Длина аудиосигнала: {audio.shape[0]}")

        input_values = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_text = processor.batch_decode(predicted_ids)[0]

        return decoded_text
    except Exception as e:
        print(f"Ошибка обработки файла {file_path}: {e}")
        return None

audio_file_path = "audio.wav"
predicted_text = transcribe_audio(audio_file_path)
if predicted_text is not None:
    print("Распознанный текст:", predicted_text)
