from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import sys

sys.path.append("src")
from ml_utils import *
from few_shot_classification import *

ml_model = joblib.load("models/ml_model.joblib")

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
        return {"prediction":ml_model.predict([clean_text])[0]}
        
    elif model_name == "camembert":
        pass
        
    elif model_name == "fewshot":
        return {"prediction":few_shot_classification(text)}
        
    else:
        raise HTTPException(status_code=400,
                           detail="Invalid model name")