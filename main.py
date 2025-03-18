import nest_asyncio
nest_asyncio.apply()
from fastapi import FastAPI, Request
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = FastAPI()

# Load the fine-tuned DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = DistilBertTokenizer.from_pretrained("./sentiment_model")
model.eval()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data['text']
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment = "positive" if prediction == 1 else "negative"
    return {"text": text, "sentiment": sentiment}

# Run the API
if __name__ == "__main__":
  import nest_asyncio
import uvicorn

nest_asyncio.apply()

uvicorn.run(app, host="127.0.0.1", port=8000)
