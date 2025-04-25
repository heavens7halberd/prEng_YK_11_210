from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_tone(text):
    try:
        result = sentiment_analyzer(text)[0]
        return {"label": result['label'], "score": result['score']}
    except Exception as e:
        return {"error": str(e)}
