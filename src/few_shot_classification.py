from transformers import (
                        pipeline, 
                        AutoTokenizer, 
                        AutoModelForCausalLM
                        )
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Create a text generator
generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
)

def few_shot_classification(text:str):

    prompt = f"""
    Tu es un expert en orientation scolaire. Ta tâche est de déterminer si un intitulé de stage de 3ᵉ est acceptable pour un collégien. Réponds uniquement par "oui" ou "non".
    
    Exemples :

    Example 1:
    Intitulé : "Stage dans une mairie"  
    Acceptable ? oui

    Example 2:
    Intitulé : "Stage chez un tatoueur"  
    Acceptable ? non

    Example 3:
    Intitulé : "Stage d’observation dans un cabinet vétérinaire"  
    Acceptable ? oui

    Example 4:
    Intitulé : "Stage dans une boîte de nuit"  
    Acceptable ? non
    
    Maintenant, analyse le nouvel intitulé :
    
    Intitulé : "{text}"  
    Acceptable ?
    """

    response =  generator(
        prompt,
        max_new_tokens=10,
        do_sample=False,
        return_full_text=False
    )[0]["generated_text"]

    if "oui" in response.lower():
        return 1
    
    return 0