from sentiment_model import load_model, preprocess

def analyze_texts(model, texts):
    results = {"pos": 0, "neg": 0, "neutral": 0}
    for text in texts:
        processed = preprocess(text)
        prediction = model.predict([processed])[0]
        results[prediction] += 1
    return results

def decision(results):
    total = results["pos"] + results["neg"]
    if total == 0: return "Not enough data"
    ratio = results["pos"] / total
    if ratio > 0.65:
        return "🟢 BUY (Greed)"
    elif ratio < 0.35:
        return "🔴 SELL (Fear)"
    else:
        return "🟡 HOLD (Neutral)"
