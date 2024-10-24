from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregando o modelo
modelo_arvore = joblib.load('C:/Users/bruno/OneDrive/Estudos/MBA Admission dataset, Class 2025/modelo_arvore.joblib')

@app.route('/')
def home():
    return "Bem-vindo à API de Previsão de Admissão!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtendo os dados da solicitação
        data = request.get_json()
        
        # Transformando os dados em um DataFrame
        df = pd.DataFrame([data])
        
        # Realizando a previsão
        prediction = modelo_arvore.predict(df)
        
        # Retornando o resultado da previsão
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
