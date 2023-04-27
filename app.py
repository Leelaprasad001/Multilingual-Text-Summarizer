from flask import Flask, render_template, request
from langdetect import detect
from googletrans import Translator
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('Bert_model')
model = EncoderDecoderModel.from_pretrained('Bert_model').to(device)

def generate_summary(text):
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length = 512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)
  


def detect_language(input_text):
    try:
        return detect(input_text)
    except:
        return "Could not detect language"
        
    
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST','GET'])
def summarize():
    if request.method=='POST': 
        dest_lng= request.form['lang']  
        input_text = request.form.get('text1')
        translator = Translator()
        translation = translator.translate(input_text, dest="en")
        text1 = generate_summary(translation.text)
        translation = translator.translate(text1, dest= dest_lng)
        output_text = translation.text
        return render_template('index.html', input_text = input_text, output_text = output_text)
    elif request.method == 'GET':
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
    