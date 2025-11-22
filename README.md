# Pose Analyzer - Aplicación Node.js

Aplicación web para análisis de poses en tiempo real usando modelos de Machine Learning entrenados con MediaPipe.

## Características

- 🎥 Análisis en tiempo real desde cámara web
- 📹 Procesamiento de videos subidos
- 🤖 Múltiples modelos ML (XGBoost, Random Forest, SVM)
- 📊 Visualización de predicciones con probabilidades
- 🎨 Interfaz moderna y responsive

## Requisitos

- Node.js 16+ 
- Python 3.8+
- npm o yarn

## Instalación

1. Instalar dependencias de Node.js:
```bash
npm install
```

2. Instalar dependencias de Python para el servicio ML:
```bash
cd ml-service
pip install -r requirements.txt
cd ..
```

## Uso

1. Iniciar el servidor:
```bash
npm start
```

2. Abrir en el navegador:
```
http://localhost:3000
```

3. Opciones:
   - **Iniciar Cámara**: Usa la cámara web para análisis en tiempo real
   - **Subir Video**: Sube un archivo de video para procesar
   - **Seleccionar Modelo**: Elige entre XGBoost, Random Forest o SVM

## Estructura del Proyecto

```
.
├── server.js              # Servidor Express
├── package.json           # Dependencias Node.js
├── ml-service/           # Servicio Python para ML
│   ├── predict.py        # Script de inferencia
│   └── requirements.txt  # Dependencias Python
├── public/               # Frontend
│   ├── index.html       # Interfaz web
│   └── app.js           # Lógica del cliente
└── models/               # Modelos entrenados (pickle)
    ├── xgb_model.pkl
    ├── rf_model.pkl
    ├── svm_model.pkl
    ├── label_encoder.pkl
    └── coord_scaler.pkl
```

## Modelos Disponibles

- **XGBoost**: Mejor precisión (99.6%)
- **Random Forest**: Buena precisión (99.5%)
- **SVM-RBF**: Precisión moderada (97.7%)

## Notas

- Los modelos están en formato pickle (Python), por lo que se requiere un servicio Python mínimo para la inferencia
- La aplicación principal está completamente en Node.js
- MediaPipe se ejecuta en el navegador usando WebAssembly

## Licencia

MIT
