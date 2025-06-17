import numpy as np
import torch
from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

app = Flask(__name__)

def load_model():
    model_name = "Qwen/Qwen3-1.7B"
    peft_path = "./qwen-title-generator-final"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    model = PeftModel.from_pretrained(base_model, peft_path)
    model = model.merge_and_unload()
    model.eval()
    
    return model, tokenizer


model, tokenizer = load_model()

def generate_title(abstract, category):
    prompt = f"Generate a {category} title for a scientific paper with the following abstract:\n\n{abstract}\n\nTitle:"

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    title = response
    title = re.sub(r'\n+', ' ', title)
    return title

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    abstract = data.get('abstract', '')
    category = data.get('category', 'standard')

    if not abstract:
        return jsonify({"error": "Abstract cannot be empty"}), 400

    try:
        title = generate_title(abstract, category)
        return jsonify({"title": title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
