import streamlit as st
import numpy as np
from transformers import BertTokenizer,BertModel
import torch
from module import SentimentClassifier
from PIL import Image
image = Image.open('img.png')
st.image(image, caption='Sentiment Analysis')
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model=SentimentClassifier(3).to(device='cpu')
    model.load_state_dict(torch.load('best_model_state.bin',map_location='cpu'))
    return tokenizer,model
tokenizer,model = get_model()
class_names = ['negative', 'neutral', 'positive']
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")
if button :
    encoded_review=tokenizer.encode_plus(user_input,
    max_length=160,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',)
    input_ids = encoded_review['input_ids'].to(device='cpu')
    attention_mask= encoded_review['attention_mask'].to(device='cpu')
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    st.write(f'Review text: {user_input}')
    st.write(f'Sentiment : {class_names[prediction]}')