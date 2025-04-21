from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Multi-Model Text Classification API")

class TextRequest(BaseModel):
    text: str
    model_name: str

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    model_name = request.model_name

    if model_name == "ml":
        pass
    elif model_name == "camembert":
        pass
    elif model_name == "fewshot":
        pass
    else:
        raise HTTPException(status_code=400,
                           detail="Invalid model name")