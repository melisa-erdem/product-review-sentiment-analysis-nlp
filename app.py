from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Flask uygulamasını başlat
app = Flask(__name__)

# Modelin ve tokenizer'ın bulunduğu klasör
model_path = "/Users/melisaerdem/Downloads/duygu_analizi_modeli"

# Model ve tokenizer'ı yükle
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Etiketlere karşılık gelen anlamlar
label_map = {
    0: "Olumsuz",
    1: "Olumlu",
    2: "Nötr"
}

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin yapma
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Kullanıcıdan gelen metni al
        input_text = request.form['text']

        # Metni tokenizer ile işleme
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Modeli kullanarak tahmin yapma
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()

        # Tahmini etikete çevir
        predicted_label = label_map[predicted_class]
        confidence = predictions[0][predicted_class].item() * 100

        # Sonucu geri gönder
        return render_template('index.html', prediction=predicted_label, confidence=f"%{confidence:.2f}", text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
