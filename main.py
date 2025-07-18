import os
import joblib
from dotenv import load_dotenv
from fetch_data import fetch_twitter_data, fetch_reddit_data, fetch_news_data
from fetch_charts import get_technical_indicators, calculate_technical_score
from sentiment_model import preprocess  

load_dotenv()
pipeline = joblib.load("fear_greed_model.joblib")

def analyze_sentiment(texts):
    greed = fear = 0
    for t in texts:
        try:
            processed = preprocess(t)  
            label = pipeline.predict([processed])[0]
            if label == "pos": greed += 1
            elif label == "neg": fear += 1
        except Exception as e:
            print(f"⚠️ Error processing text: {str(e)}")
            continue
            
    total = greed + fear
    if total == 0: 
        print("⚠️ No valid sentiment data could be processed")
        return 50  #returning neutral if no score 
    return round((greed / total) * 100)

def generate_final_decision(tech_score, sentiment_score):
    #i made this 70% charts and 30% news but this can be tuned according to choice to counter fake news etc
    combined_score = (tech_score * 0.7) + (sentiment_score * 0.3)
    
    if combined_score > 70:
        decision = "🟢 STRONG BUY"
    elif combined_score > 60:
        decision = "🟢 BUY"
    elif combined_score > 45:
        decision = "🟡 HOLD (Mild Bullish)"
    elif combined_score > 35:
        decision = "🟡 HOLD (Mild Bearish)"
    elif combined_score > 25:
        decision = "🔴 SELL"
    else:
        decision = "🔴 STRONG SELL"
    
    return decision, combined_score

if __name__ == "__main__":
    query = "microsoft"
    ticker = "MSFT"
    
    #  fetching the charts data
    try:
        indicators = get_technical_indicators(ticker)
        tech_score = calculate_technical_score(indicators)
        print("\n📈 Technical Indicators:")
        print(f"Current Price: {indicators['current_price']:.2f}")
        print(f"50-Day MA: {indicators['ma_50']:.2f} | 200-Day MA: {indicators['ma_200']:.2f}")
        print(f"RSI: {indicators['rsi']:.2f} | MACD Hist: {indicators['macd_hist']:.2f}")
        print(f"Volume Spike: {'Yes' if indicators['volume_spike'] else 'No'}")
        print(f"Technical Score: {tech_score}/100")
    except Exception as e:
        print(f"⚠️ Technical analysis failed: {str(e)}")
        tech_score = 50  # Neutral if error
    
    # Fetching the sentiment analysis 
    try:
        # tweets = fetch_twitter_data(query, 10)
        reddit = fetch_reddit_data(["stocks", "wallstreetbets"], query, 10)
        news = fetch_news_data(query, n_news=10)
        all_data =   reddit + news
        
        sentiment_score = analyze_sentiment(all_data)
        print(f"\n📊 Sentiment Analysis: {sentiment_score}/100")
        print(f"  Greed/Fear Ratio: {sentiment_score}:{100-sentiment_score}")
        
        # final decision 
        decision, combined_score = generate_final_decision(tech_score, sentiment_score)
        print(f"\n💡 Final Decision (Score: {combined_score:.1f}/100): {decision}")
        
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed: {str(e)}")
        print(f"💡 Technical-only Decision: {'🟢 BUY' if tech_score > 50 else '🔴 SELL'} (Score: {tech_score}/100)")
