from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

#  CONFIGURACIÓN DEL MESERO (API) 
app = Flask(__name__)
CORS(app) # Permite que cualquiera (tu App) haga pedidos

# CARGAR AL CHEF (TUS ARCHIVOS)
print("Cargando el cerebrazo")
modelo = joblib.load('modelo_cultivos_rf.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
print("Cerebelo listo")

#DICCIONARIO DE TRADUCCIÓN (TIPO DE TIERRA A NPK) 
# Esto convierte lo que el usuario ve en lo que el modelo necesita
suelo_mapping = {
    "Arcillosa": {"N": 60.0, "P": 45.0, "K": 35.0, "ph": 7.5},
    "Arenosa":   {"N": 20.0, "P": 15.0, "K": 25.0, "ph": 6.0},
    "Limosa":    {"N": 80.0, "P": 50.0, "K": 50.0, "ph": 6.8},
    "Negra":     {"N": 100.0, "P": 60.0, "K": 50.0, "ph": 6.5},
    "Roja":      {"N": 40.0, "P": 55.0, "K": 40.0, "ph": 5.5}
}

# ENDPOINTS DE LA API
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # 1. Recibir la orden del cliente (JSON)
        datos = request.get_json()
        # Se espera algo como: "tierra": "Arcillosa", "temp": 25, "hum": 60, "lluvia": 100
        tierra = datos['tierra']
        temp = float(datos['temp'])
        hum = float(datos['hum'])
        lluvia = float(datos['lluvia'])

        # 2. Traducir la "Tierra" a N, P, K, pH
        if tierra in suelo_mapping:
            nutrientes = suelo_mapping[tierra]
            N = nutrientes['N']
            P = nutrientes['P']
            K = nutrientes['K']
            ph = nutrientes['ph']
        else:
            return jsonify({'error': 'Tipo de tierra no reconocido'}), 400

        # 3. Preparar los datos para el modelo
        # IMPORTANTE: El orden debe ser IGUAL al del entrenamiento
        # [N, P, K, temperature, humidity, ph, rainfall]
        features = np.array([[N, P, K, temp, hum, ph, lluvia]])

        # 4. Ajustar la escala (Scaler)
        features_scaled = scaler.transform(features)

        # 5. El Chef cocina (Predicción)
        prediccion_num = modelo.predict(features_scaled)
        
        # 6. Traducir el resultado (Número -> Nombre del cultivo)
        resultado_texto = label_encoder.inverse_transform(prediccion_num)[0]

        # 7. Entregar el platillo al cliente
        return jsonify({
            'cultivo_ideal': resultado_texto,
            'mensaje': f'Para tierra {tierra} y clima actual, te recomiendo: {resultado_texto}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- ENCENDER EL RESTAURANTE ---
if __name__ == '__main__':
    # host='0.0.0.0' significa que es visible en tu red wifi local
    app.run(host='0.0.0.0', port=5000, debug=True)