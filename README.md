# Textual classification of internship titles

Development of an API to classify internships (internship accepted or not by the school) given their titles. 3 approaches:
- Machine Learning : static embedding + classification algorithm
- Classification with CamemBERT
- Zero-Shot and Few-Shot Learning using a generative model

To install the necessary libraries:
```bash
pip install uv
uv pip install -r requirements.txt
```
For the Zero-Shot and Few-Shot learning approches you need to provide a HuggingFace token for the model used after doing:
```bash
huggingface-cli login
```
