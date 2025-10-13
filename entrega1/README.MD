# **Preguntas de interés**

1. ¿Puede un modelo de inteligencia artificial clasificar correctamente en tiempo real las actividades **caminar hacia la cámara**, **caminar de regreso**, **girar**, **sentarse** y **ponerse de pie**, a partir de los landmarks articulares obtenidos con *MediaPipe Pose*?
2. ¿Qué características (ángulos, velocidades, inclinaciones) son más determinantes para distinguir entre las actividades, considerando diferencias entre personas y condiciones de grabación?
3. ¿Qué tan robusto puede ser el modelo frente a variaciones de distancia, ángulo de cámara, iluminación o velocidad de ejecución?
4. ¿Cómo puede esta herramienta ser aplicada en entornos de **ergonomía laboral**, **deporte** o **rehabilitación física** para automatizar el análisis postural?

# **Tipo de problema**

Este proyecto aborda un **problema de clasificación supervisada multiclase** dentro del campo de **computer vision**.
Cada clip de video contiene una única actividad, y el objetivo del modelo es asignar una etiqueta entre las cinco clases posibles.
El modelo aprenderá a partir de características derivadas de las coordenadas tridimensionales (x, y, z) de las articulaciones detectadas con **MediaPipe Pose**.

# **Metodología**

**A. Adquisición y formato**

* **Resolución/FPS**: 1920×1080 @ **30 FPS** (mín. 1280×720).
* **Cámara**: trípode a **1.40 m** de altura, **distancia 3.0 m** al sujeto, **vista frontal 0°**.
* **Clips**: **1 actividad/clip**, duración **5–8 s** (turn: 3–5 s).
* **Clases**: `walk_fwd`, `walk_bwd`, `turn`, `sit`, `stand`.
* **Nombres**: `YYYYMMDD_SUB{nn}_VIEWF_CLS-{class}_REP{r}.mp4`.
* **Metadatos sidecar** (`.json` por clip):

  ```json
  {
    "file":"20251012_SUB03_VIEWF_CLS-sit_REP2.mp4",
    "subject_id":"SUB03","class":"sit","view":"F","rep":2,
    "fps":30,"resolution":"1920x1080","distance_m":3.0,
    "lighting":"uniform","notes":"full body visible"
  }
  ```

* **Estructura**:

  ```
  dataset/
    raw/           # .mp4
    metadata/      # .json sidecar
    annotations/   # export LabelStudio
    splits/        # train/val/test .txt
  ```

**B. Anotación**

* Herramienta: **LabelStudio** (modo clasificación por clip).
* Validación: **doble rotulado** 10% de clips; **acuerdo ≥0.9** (Cohen’s κ).

**C. Extracción de landmarks y preprocesamiento**

* **Detector**: *MediaPipe Pose* (33 landmarks, x,y,[z], visibility).
* **Interpolación** lineal si **gaps ≤3 frames**.
* **Normalización espacial**:

  * Trasladar al **centro de pelvis** (promedio caderas).
  * Escalar por **distancia hombros–caderas** (altura de tronco) o **cadera-cadera** (ancho pélvico).
* **Normalización temporal**: **resample** a **30 FPS**.
* **Filtrado**: **Savitzky–Golay** (ventana = 7, orden = 2) por landmark.
* **Estandarización** de features: **z-score** con estadísticas de *train*.

**D. Ingeniería de características (por ventana 0.5–1.0 s con *hop* 0.25 s)**

* **Ángulos** (grados):

  * `hip_knee_ankle` (izq/der), `shoulder_hip_knee`, `shoulder_line_vs_hip_line`.
* **Inclinación de tronco**: ángulo entre línea hombros y horizontal; **roll** lateral y **pitch**.
* **Cinemática**:

  * Velocidad y **varianza** de tobillos y caderas (norma Δpos/frame).
  * **Cadencia** (picos cíclicos en tobillos) para *walk_fwd/bwd*.
* **Estadísticos por ventana**: mean, std, p10/p90, IQR, max-min.
* **Dimensión objetivo**: **≤128 features** (PCA opcional a 32–64 si es necesario).

**E. Partición y validación**

* **Split por sujeto** (sin fuga): **70/15/15** train/val/test.
* **Semilla** fija (p. ej., 42).
* **Early stopping** con **macro-F1** en *val* (paciencia = 10).

**F. Modelado**

* **SVM-RBF**: `C∈{1,10,100}`, `gamma∈{1e-3,1e-2,1e-1}`.
* **RandomForest**: `n_estimators=300`, `max_depth∈{12,18,None}`, `min_samples_leaf∈{1,3,5}`.
* **XGBoost**: `n_estimators=600`, `max_depth∈{6,8}`, `eta∈{0.03,0.1}`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`.
* **Selección** por **macro-F1 (val)**; desempate por **macro-Recall**.

**G. Inferencia en tiempo real (definición)**

* **Buffer** deslizante **30 frames** (1 s), hop **7–10 frames**.
* **Suavizado** de predicción: **voto mayoritario** últimas **3** ventanas.
* **Latencia objetivo**: **≤200 ms** fuera de captura.

# **Métricas**
# **Métricas**

| Métrica       | Descripción                                                 | Objetivo                            |
| ------------- | ----------------------------------------------------------- | ----------------------------------- |
| **Accuracy**  | Porcentaje total de actividades correctamente clasificadas. | Medir efectividad general.          |
| **Precision** | Razón entre verdaderos positivos y positivos predichos.     | Evitar falsas detecciones.          |
| **Recall**    | Razón entre verdaderos positivos y positivos reales.        | Garantizar cobertura.               |
| **F1-Score**  | Media armónica entre precision y recall.                    | Balancear precisión y sensibilidad. |


# **Datos recolectados**

* **Cantidad inicial:** 150 clips (10 sujetos × 5 actividades × 3 repeticiones).
* **Formato:** `.mp4`, 1080p, 30 FPS.
* **Duración por clip:** 5–8 segundos.
* **Actividades:** `walk_fwd`, `walk_bwd`, `turn`, `sit`, `stand`.
* **Etiquetado:** con **LabelStudio**.
* **Estructura del dataset:**

  ```
  dataset/
    raw/
    annotations/
    metadata/
    splits/
  ```

# **Análisis exploratorio de los datos**

Ejecutar en la primera iteración (sobre 150 clips):

1. **Distribución por clase** (clips, frames) y por **sujeto**.
2. **% de detección** de landmarks por frame (≥ 95% esperado).
3. **Estadísticos** de ángulos clave (media, std por clase).
4. **Velocidades**: histograma de velocidad de tobillos y caderas por clase.
5. **Separabilidad**: PCA/UMAP 2D de features coloreado por clase.
6. **Ruido**: tasa de outliers (>3σ) en tobillos durante *turn*; justificación del filtro.
7. **Baseline** rápido: Regresión logística con Macro-F1 esperado **≥ 0.60** (sanity check).

## **Estrategia de obtención de datos**

Con el fin de de tener un plan dado el caso de que los resultados sean insuficientes, se optará por:

1. Consultar por bases de datos que tengan o solucionen un problema similar.
2. Recurrir a proyectos anteriores realizados por estudiantes de la universidad Icesi
3. Creación de muestras por medio de la grabación de múltiples individuos colaboradores que se presten su imagen con fines de educación

# **Siguientes pasos**

1. **Cierre de captura** (150 clips) y **QA** (checklist de visibilidad, nombre, metadatos).
2. **Extracción landmarks** + **features** en *batch*; guardado en **parquet/csv**.
3. **Splits por sujeto** + *grid search* de SVM/RF/XGB en **val**.
4. **Reporte de métricas** (macro-F1, por clase, confusión) y **error analysis** (fallos por clase).
5. **Ajustes** (features/hiperparámetros) según error analysis.
6. **Prototipo** de inferencia en tiempo real (buffer 1 s + voto 3 ventanas).