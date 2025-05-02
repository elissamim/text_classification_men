from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
                         CamembertTokenizer,
                         CamembertForSequenceClassification
                         )
import joblib
import sys
import torch
import os
import s3fs
from dotenv import load_dotenv
import tempfile

sys.path.append("src")
from ml_utils import *
from few_shot_classification import *

load_dotenv()

# Load ML model
ml_model = joblib.load("models/ml_model.joblib")

# Load Camembert Model
s3_model_path = "elissamim/text_classification_men/models/camembert_model/"
fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
            key=os.environ["Accesskey"],
            secret=os.environ["Secretkey"],
            token=os.environ["Token"]
)

with tempfile.TemporaryDirectory() as tmp_dir:

    for file in fs.ls(s3_model_path):
        file_name = os.path.basename(file)
        local_path = os.path.join(tmp_dir, file_name)
        fs.get(file, local_path)

    # Load model and tokenizer from the local temp directory
    camembert_tokenizer = CamembertTokenizer.from_pretrained(tmp_dir)
    camembert_model = CamembertForSequenceClassification.from_pretrained(tmp_dir)
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
        return {"prediction_ml":int(ml_model.predict([clean_text])[0])}
        
    elif model_name == "camembert":

        inputs = camembert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = camembert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).item()

        return {"prediction_camembert":int(predicted_label)}
        
    elif model_name == "fewshot":
        return {"prediction_fewshot":int(few_shot_classification(text))}
        
    else:
        raise HTTPException(status_code=400,
                           detail="Invalid model name")