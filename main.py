from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
                         CamembertTokenizer,
                         CamembertForSequenceClassification
                         )
import joblib
import sys
import torch

sys.path.append("src")
from ml_utils import *
from few_shot_classification import *

# Load ML model
ml_model = joblib.load("models/ml_model.joblib")

# Load Camembert Model
camembert_model = CamembertForSequenceClassification.from_pretrained("models/camembert_model")
camembert_tokenizer = CamembertTokenizer.from_pretrained("models/camembert_model")
camembert_model.eval()

app = FastAPI(title="Multi-Model Text Classification API")

class TextRequest(BaseModel):
    
    text: str
    model_name: str

@app.post("/predict")
def predict(request: TextRequest):
    
    text = request.text
    model_name = request.model_name

    if model_name == "ml":
        
        clean_text = nltk_text_preprocessing(text)
        return {"prediction_ml":ml_model.predict([clean_text])[0]}
        
    elif model_name == "camembert":

        inputs = camembert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = camembert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).item()

        return {"prediction_camembert":int(predicted_label)}
        
    elif model_name == "fewshot":
        return {"prediction_fewshot":few_shot_classification(text)}
        
    else:
        raise HTTPException(status_code=400,
                           detail="Invalid model name")