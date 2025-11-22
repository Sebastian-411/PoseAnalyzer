import os
import sys
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.signal import savgol_filter
from threading import Thread
import time
import math

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
NUM_LANDMARKS = len(list(mp_pose.PoseLandmark))

class PoseAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Analyzer - Análisis de Actividades")
        self.root.geometry("900x700")
        
        # Variables
        self.video_path = None
        self.models = {}
        self.label_encoder = None
        self.coord_scaler = None
        self.processing = False
        
        # Variables para webcam
        self.webcam_active = False
        self.cap = None
        self.frame_buffer = []  # Buffer para acumular frames
        self.buffer_size = 30  # Tamaño del buffer (ventana)
        self.process_interval = 10  # Procesar cada N frames
        self.frame_count = 0
        self.webcam_thread = None
        self.pose_detector = None
        self.current_prediction = None  # Para mostrar predicción en el video
        
        # Crear interfaz primero
        self.create_ui()
        
        # Cargar modelos después de crear la UI
        self.load_models()
        
        # Configurar cierre limpio
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_models(self):
        """Cargar todos los modelos y preprocesadores"""
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        try:
            # Cargar modelos
            with open(os.path.join(models_dir, 'rf_model.pkl'), 'rb') as f:
                self.models['RandomForest'] = pickle.load(f)
            
            with open(os.path.join(models_dir, 'svm_model.pkl'), 'rb') as f:
                self.models['SVM'] = pickle.load(f)
            
            with open(os.path.join(models_dir, 'xgb_model.pkl'), 'rb') as f:
                self.models['XGBoost'] = pickle.load(f)
            
            # Cargar encoder y scaler
            with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(os.path.join(models_dir, 'coord_scaler.pkl'), 'rb') as f:
                self.coord_scaler = pickle.load(f)
            
            # Obtener el orden de columnas esperado desde los modelos
            # Después de la agrupación, las columnas tienen sufijos como _mean, _std, etc.
            self.expected_feature_order = None
            
            # XGBoost con feature_names_in_
            if hasattr(self.models['XGBoost'], 'feature_names_in_'):
                self.expected_feature_order = list(self.models['XGBoost'].feature_names_in_)
            # XGBoost antiguo con get_booster
            elif hasattr(self.models['XGBoost'], 'get_booster'):
                try:
                    booster = self.models['XGBoost'].get_booster()
                    if hasattr(booster, 'feature_names'):
                        self.expected_feature_order = list(booster.feature_names)
                except:
                    pass
            
            # Si no se pudo obtener de XGBoost, intentar desde RandomForest
            if self.expected_feature_order is None:
                if hasattr(self.models['RandomForest'], 'feature_names_in_'):
                    self.expected_feature_order = list(self.models['RandomForest'].feature_names_in_)
            
            # Si aún no se tiene, intentar desde SVM
            if self.expected_feature_order is None:
                if hasattr(self.models['SVM'], 'named_steps'):
                    # SVM puede estar en un Pipeline
                    svm_step = self.models['SVM'].named_steps.get('svc', None)
                    if svm_step and hasattr(svm_step, 'feature_names_in_'):
                        self.expected_feature_order = list(svm_step.feature_names_in_)
            
            # Si aún no se tiene, el orden se determinará dinámicamente
            # basándose en las columnas generadas por la agrupación
            if self.expected_feature_order is None:
                self.expected_feature_order = None  # Se determinará dinámicamente
            
            self.status_label.config(text="Modelos cargados correctamente", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron cargar los modelos:\n{str(e)}")
            self.status_label.config(text="Error al cargar modelos", fg="red")
            self.expected_feature_order = None
    
    def create_ui(self):
        """Crear la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, text="Pose Analyzer", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Botón para seleccionar video
        select_btn = ttk.Button(main_frame, text="Seleccionar Video", command=self.select_video)
        select_btn.grid(row=1, column=0, pady=10, padx=5, sticky=(tk.W, tk.E))
        
        # Botón para procesar
        process_btn = ttk.Button(main_frame, text="Procesar Video", command=self.process_video, state="disabled")
        process_btn.grid(row=1, column=1, pady=10, padx=5, sticky=(tk.W, tk.E))
        self.process_btn = process_btn
        
        # Botón para webcam
        self.webcam_btn = ttk.Button(main_frame, text="Iniciar Webcam", command=self.toggle_webcam)
        self.webcam_btn.grid(row=2, column=0, columnspan=2, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        # Label para mostrar video seleccionado
        self.video_label = ttk.Label(main_frame, text="Ningún video seleccionado", foreground="gray")
        self.video_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Status label (usar tk.Label para soportar fg)
        self.status_label = tk.Label(main_frame, text="Listo", fg="blue")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Frame para resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de los Modelos", padding="10")
        results_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Área de texto para resultados
        self.results_text = tk.Text(results_frame, height=15, width=80, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para resultados
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def select_video(self):
        """Seleccionar archivo de video"""
        filetypes = [
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.MOV"),
            ("Todos los archivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Seleccionar Video",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path = filename
            self.video_label.config(text=f"Video: {os.path.basename(filename)}", foreground="black")
            self.process_btn.config(state="normal")
            self.results_text.delete(1.0, tk.END)
    
    def process_video(self):
        """Procesar el video en un hilo separado"""
        if not self.video_path or self.processing:
            return
        
        self.processing = True
        self.process_btn.config(state="disabled")
        self.progress.start()
        self.status_label.config(text="Procesando video...", fg="blue")
        self.results_text.delete(1.0, tk.END)
        
        # Ejecutar en hilo separado para no bloquear la UI
        thread = Thread(target=self._process_video_thread)
        thread.daemon = True
        thread.start()
    
    def _process_video_thread(self):
        """Procesar video en hilo separado"""
        try:
            # Extraer landmarks del video
            landmarks_data = self.extract_landmarks()
            
            if not landmarks_data:
                self.root.after(0, lambda: self._show_error("No se detectaron poses en el video"))
                return
            
            # Preprocesar datos
            processed_data = self.preprocess_data(landmarks_data)
            
            # Hacer predicciones
            predictions = self.predict_all_models(processed_data)
            
            # Mostrar resultados
            self.root.after(0, lambda: self._show_results(predictions, landmarks_data))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Error al procesar: {str(e)}"))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_label.config(text="Procesamiento completado", fg="green"))
    
    def extract_landmarks(self):
        """Extraer landmarks del video usando MediaPipe y mostrar visualización"""
        landmarks_list = []
        window_name = "Pose Analyzer - Video"
        
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("No se pudo abrir el video")
            
            # Obtener FPS del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Default si no se puede obtener
            
            frame_delay = int(1000 / fps)  # Delay en milisegundos
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Dibujar landmarks en el frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_data = {}
                    
                    # Extraer landmarks exactamente como en MediaPipe.py
                    for i in range(NUM_LANDMARKS):
                        lm = results.pose_landmarks.landmark[i]
                        frame_data[f'x_{i}'] = lm.x
                        frame_data[f'y_{i}'] = lm.y
                        frame_data[f'z_{i}'] = lm.z
                        frame_data[f'visibility_{i}'] = lm.visibility
                    
                    landmarks_list.append(frame_data)
                
                # Mostrar información en el frame
                progress_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(frame, progress_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Mostrar el frame
                cv2.imshow(window_name, frame)
                
                # Salir si se presiona 'q' o ESC
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q') or key == 27:  # 'q' o ESC
                    break
                
                # Verificar si la ventana fue cerrada
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                frame_count += 1
                
                # Actualizar progreso cada 10 frames
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.root.after(0, lambda p=progress: self.status_label.config(
                        text=f"Extrayendo landmarks... {p:.1f}%", fg="blue"))
            
            cap.release()
            cv2.destroyWindow(window_name)
        
        return landmarks_list
    
    def create_overlapping_windows(self, df, window_size=30, hop_size=10):
        """
        Crea ventanas deslizantes con redundancia (overlapping).
        Cada ventana contiene múltiples frames agregados estadísticamente.
        Adaptado para un solo video sin label.
        """
        grouped_windows = []
        
        # Ordenar por índice (que representa el orden temporal)
        df = df.sort_index().reset_index(drop=True)
        num_frames = len(df)
        
        # Si no hay suficientes frames para una ventana, devolver el dataframe original
        if num_frames < window_size:
            # Si hay muy pocos frames, crear una sola ventana con todos los frames
            window = df.copy()
            window_record = {}
            
            numeric_cols = window.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'frame_id']
            
            for col in numeric_cols:
                values = window[col].values
                window_record[f'{col}_mean'] = np.mean(values)
                window_record[f'{col}_std'] = np.std(values) if len(values) > 1 else 0
                window_record[f'{col}_min'] = np.min(values)
                window_record[f'{col}_max'] = np.max(values)
                window_record[f'{col}_median'] = np.median(values)
                window_record[f'{col}_p25'] = np.percentile(values, 25)
                window_record[f'{col}_p75'] = np.percentile(values, 75)
            
            grouped_windows.append(window_record)
        else:
            # Crear ventanas deslizantes
            for start in range(0, num_frames - window_size + 1, hop_size):
                end = start + window_size
                window = df.iloc[start:end].copy()
                
                # Crear registro agregado para esta ventana
                window_record = {}
                
                # Obtener todas las columnas numéricas (excepto frame_id)
                numeric_cols = window.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'frame_id']
                
                # Calcular estadísticas para cada feature
                for col in numeric_cols:
                    values = window[col].values
                    window_record[f'{col}_mean'] = np.mean(values)
                    window_record[f'{col}_std'] = np.std(values) if len(values) > 1 else 0
                    window_record[f'{col}_min'] = np.min(values)
                    window_record[f'{col}_max'] = np.max(values)
                    window_record[f'{col}_median'] = np.median(values)
                    window_record[f'{col}_p25'] = np.percentile(values, 25)
                    window_record[f'{col}_p75'] = np.percentile(values, 75)
                
                grouped_windows.append(window_record)
        
        return pd.DataFrame(grouped_windows)
    
    def preprocess_data(self, landmarks_data):
        """Aplicar el mismo preprocesamiento que en el notebook con agrupación de ventanas"""
        if not landmarks_data:
            return None
        
        df = pd.DataFrame(landmarks_data)
        
        # Remover columnas de visibility
        columns_to_remove = [col for col in df.columns if 'visibility' in col]
        df = df.drop(columns=columns_to_remove)
        
        # Normalizar coordenadas primero (antes de calcular features derivadas)
        coord_cols = [f'{axis}_{i}' for i in range(NUM_LANDMARKS) for axis in ['x', 'y', 'z']]
        available_coords = [col for col in coord_cols if col in df.columns]
        
        if available_coords and self.coord_scaler is not None:
            try:
                df[available_coords] = self.coord_scaler.transform(df[available_coords])
            except Exception as e:
                # Si el scaler espera más columnas, intentar solo con las disponibles
                print(f"Advertencia al normalizar: {e}")
        
        # Filtrado suave (Savitzky-Golay)
        if len(df) >= 5:
            window_size = 5
            poly_order = 2
            for col in coord_cols:
                if col in df.columns:
                    try:
                        df[col] = savgol_filter(df[col], window_size, poly_order)
                    except:
                        pass
        elif len(df) >= 3:
            window_size = 3
            poly_order = 1
            for col in coord_cols:
                if col in df.columns:
                    try:
                        df[col] = savgol_filter(df[col], window_size, poly_order)
                    except:
                        pass
        
        # Calcular velocidades
        for i in range(NUM_LANDMARKS):
            for axis in ['x', 'y', 'z']:
                col = f'{axis}_{i}'
                if col in df.columns:
                    df[f'vel_{axis}_{i}'] = df[col].diff().fillna(0)
        
        # Calcular ángulo del brazo derecho
        angles = []
        for idx in range(len(df)):
            try:
                a = np.array([df.loc[idx, f'{axis}_11'] for axis in ['x', 'y', 'z'] if f'{axis}_11' in df.columns])
                b = np.array([df.loc[idx, f'{axis}_13'] for axis in ['x', 'y', 'z'] if f'{axis}_13' in df.columns])
                c = np.array([df.loc[idx, f'{axis}_15'] for axis in ['x', 'y', 'z'] if f'{axis}_15' in df.columns])
                
                if len(a) == 3 and len(b) == 3 and len(c) == 3:
                    ba = a - b
                    bc = c - b
                    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                    angles.append(angle)
                else:
                    angles.append(0)
            except:
                angles.append(0)
        
        df['angle_brazo_derecho'] = angles
        
        # Calcular inclinación del tronco
        try:
            df['inclinacion_tronco'] = np.degrees(np.arctan2(
                (df['y_24'] + df['y_23']) / 2 - (df['y_12'] + df['y_11']) / 2,
                (df['z_24'] + df['z_23']) / 2 - (df['z_12'] + df['z_11']) / 2
            ))
        except:
            df['inclinacion_tronco'] = 0
        
        # Agregar frame_id
        df['frame_id'] = range(len(df))
        
        # AGRUPAR EN VENTANAS DESLIZANTES (igual que en el notebook)
        WINDOW_SIZE = 30  # 30 frames = 1 segundo a 30 FPS
        HOP_SIZE = 10     # Paso de 10 frames (redundancia: 20 frames solapados)
        
        df = self.create_overlapping_windows(df, window_size=WINDOW_SIZE, hop_size=HOP_SIZE)
        
        # Asegurar que las columnas estén en el orden esperado por los modelos
        if hasattr(self, 'expected_feature_order') and self.expected_feature_order:
            expected_cols = self.expected_feature_order
            # Agregar columnas faltantes con valor 0
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Filtrar solo las columnas esperadas
            return df[expected_cols]
        else:
            # Si no hay orden esperado, devolver todas las columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return df[numeric_cols]
    
    def predict_all_models(self, processed_data):
        """Hacer predicciones con todos los modelos"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Obtener el orden de columnas esperado
                expected_cols = None
                
                # Intentar obtener desde el modelo
                if hasattr(model, 'feature_names_in_'):
                    expected_cols = list(model.feature_names_in_)
                elif model_name == 'XGBoost' and hasattr(model, 'get_booster'):
                    try:
                        expected_cols = model.get_booster().feature_names
                    except:
                        pass
                elif model_name == 'SVM' and hasattr(model, 'named_steps'):
                    # SVM puede estar en un Pipeline
                    svm_step = model.named_steps.get('svc', None)
                    if svm_step and hasattr(svm_step, 'feature_names_in_'):
                        expected_cols = list(svm_step.feature_names_in_)
                
                # Si no se pudo obtener, usar el orden esperado global
                if expected_cols is None and hasattr(self, 'expected_feature_order') and self.expected_feature_order:
                    expected_cols = self.expected_feature_order
                
                # Preparar los datos con las columnas esperadas
                if expected_cols:
                    # Agregar columnas faltantes con valor 0
                    missing_cols = [col for col in expected_cols if col not in processed_data.columns]
                    for col in missing_cols:
                        processed_data[col] = 0
                    
                    # Reordenar columnas al orden esperado
                    processed_data = processed_data[expected_cols]
                
                # Hacer predicciones
                frame_predictions = model.predict(processed_data)
                
                labels = self.label_encoder.inverse_transform(frame_predictions)
                
                unique, counts = np.unique(labels, return_counts=True)
                freq_dict = dict(zip(unique, counts))
                total = len(labels)
                
                predictions[model_name] = {
                    'labels': labels,
                    'frequencies': freq_dict,
                    'total_frames': total,
                    'most_common': max(freq_dict.items(), key=lambda x: x[1]) if freq_dict else None
                }
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                predictions[model_name] = {'error': error_msg}
        
        return predictions
    
    def _show_results(self, predictions, landmarks_data):
        """Mostrar resultados en la interfaz"""
        self.results_text.delete(1.0, tk.END)
        
        # Encabezado
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, "RESULTADOS DEL ANÁLISIS DE POSES\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")
        
        # Calcular número de ventanas procesadas (no frames individuales)
        total_windows = 0
        for model_name, pred_data in predictions.items():
            if 'total_frames' in pred_data:
                total_windows = pred_data['total_frames']
                break
        
        self.results_text.insert(tk.END, f"Total de frames extraídos: {len(landmarks_data)}\n")
        self.results_text.insert(tk.END, f"Total de ventanas procesadas: {total_windows}\n")
        self.results_text.insert(tk.END, f"(Ventanas de 30 frames con paso de 10 frames)\n\n")
        
        # Resultados por modelo
        for model_name, pred_data in predictions.items():
            self.results_text.insert(tk.END, f"\n{'=' * 70}\n")
            self.results_text.insert(tk.END, f"MODELO: {model_name}\n")
            self.results_text.insert(tk.END, f"{'=' * 70}\n\n")
            
            if 'error' in pred_data:
                self.results_text.insert(tk.END, f"ERROR: {pred_data['error']}\n\n")
                continue
            
            # Predicción más común
            if pred_data['most_common']:
                activity, count = pred_data['most_common']
                percentage = (count / pred_data['total_frames']) * 100
                self.results_text.insert(tk.END, f"Actividad más probable: {activity}\n")
                self.results_text.insert(tk.END, f"Confianza: {percentage:.1f}% ({count}/{pred_data['total_frames']} frames)\n\n")
            
            # Distribución completa
            self.results_text.insert(tk.END, "Distribución de actividades:\n")
            self.results_text.insert(tk.END, "-" * 70 + "\n")
            
            # Ordenar por frecuencia
            sorted_freq = sorted(pred_data['frequencies'].items(), key=lambda x: x[1], reverse=True)
            for activity, count in sorted_freq:
                percentage = (count / pred_data['total_frames']) * 100
                self.results_text.insert(tk.END, f"  {activity:20s}: {percentage:5.1f}% ({count:4d} frames)\n")
            
            self.results_text.insert(tk.END, "\n")
        
        # Comparación entre modelos
        self.results_text.insert(tk.END, f"\n{'=' * 70}\n")
        self.results_text.insert(tk.END, "COMPARACIÓN ENTRE MODELOS\n")
        self.results_text.insert(tk.END, f"{'=' * 70}\n\n")
        
        for model_name, pred_data in predictions.items():
            if 'most_common' in pred_data and pred_data['most_common']:
                activity, count = pred_data['most_common']
                percentage = (count / pred_data['total_frames']) * 100
                self.results_text.insert(tk.END, f"{model_name:15s}: {activity:20s} ({percentage:5.1f}%)\n")
        
        self.results_text.see(1.0)  # Scroll al inicio
    
    def _show_error(self, message):
        """Mostrar mensaje de error"""
        messagebox.showerror("Error", message)
        self.status_label.config(text="Error", fg="red")
    
    def toggle_webcam(self):
        """Iniciar o detener la webcam"""
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Iniciar captura de webcam"""
        try:
            # Intentar abrir la webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la webcam")
                return
            
            # Configurar resolución (opcional)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Inicializar detector de poses
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Inicializar variables
            self.webcam_active = True
            self.frame_buffer = []
            self.frame_count = 0
            
            # Actualizar UI
            self.webcam_btn.config(text="Detener Webcam")
            self.process_btn.config(state="disabled")
            self.status_label.config(text="Webcam activa - Analizando en tiempo real...", fg="green")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=" * 70 + "\n")
            self.results_text.insert(tk.END, "ANÁLISIS EN TIEMPO REAL - WEBCAM\n")
            self.results_text.insert(tk.END, "=" * 70 + "\n\n")
            self.results_text.insert(tk.END, "📹 Webcam activada\n")
            self.results_text.insert(tk.END, "⏳ Acumulando frames para análisis...\n")
            self.results_text.insert(tk.END, f"📊 Se necesitan {self.buffer_size} frames para el primer análisis.\n")
            self.results_text.insert(tk.END, f"🔄 El análisis se actualizará cada {self.process_interval} frames.\n\n")
            self.results_text.insert(tk.END, "💡 Instrucciones:\n")
            self.results_text.insert(tk.END, "   - Asegúrate de estar frente a la cámara\n")
            self.results_text.insert(tk.END, "   - Realiza las actividades: caminar, girar, sentarse, pararse\n")
            self.results_text.insert(tk.END, "   - Los resultados aparecerán automáticamente\n\n")
            
            # Iniciar hilo de captura
            self.webcam_thread = Thread(target=self.webcam_capture_loop)
            self.webcam_thread.daemon = True
            self.webcam_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar webcam:\n{str(e)}")
            self.webcam_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def stop_webcam(self):
        """Detener captura de webcam"""
        self.webcam_active = False
        
        # Cerrar todas las ventanas de OpenCV
        cv2.destroyAllWindows()
        
        # Liberar recursos
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.pose_detector:
            self.pose_detector.close()
            self.pose_detector = None
        
        # Actualizar UI
        self.webcam_btn.config(text="Iniciar Webcam")
        self.process_btn.config(state="normal" if self.video_path else "disabled")
        self.status_label.config(text="Webcam detenida", fg="blue")
        self.frame_buffer = []
        self.frame_count = 0
        self.current_prediction = None
    
    def webcam_capture_loop(self):
        """Loop principal para capturar y procesar frames de la webcam"""
        window_name = "Pose Analyzer - Webcam"
        
        while self.webcam_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Convertir a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar pose
            if self.pose_detector:
                results = self.pose_detector.process(frame_rgb)
                
                # Dibujar landmarks en el frame
                if results.pose_landmarks:
                    # Dibujar conexiones y landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Extraer landmarks del frame
                    frame_data = {}
                    for i in range(NUM_LANDMARKS):
                        lm = results.pose_landmarks.landmark[i]
                        frame_data[f'x_{i}'] = lm.x
                        frame_data[f'y_{i}'] = lm.y
                        frame_data[f'z_{i}'] = lm.z
                        frame_data[f'visibility_{i}'] = lm.visibility
                    
                    # Agregar al buffer
                    self.frame_buffer.append(frame_data)
                    self.frame_count += 1
                    
                    # Mantener solo los últimos buffer_size frames
                    if len(self.frame_buffer) > self.buffer_size:
                        self.frame_buffer.pop(0)
                    
                    # Procesar cada process_interval frames o cuando tengamos suficientes frames
                    if len(self.frame_buffer) >= self.buffer_size and self.frame_count % self.process_interval == 0:
                        self.process_realtime_buffer()
                
                # Mostrar predicción actual en el frame
                if self.current_prediction:
                    # Obtener la predicción más común
                    pred_text = "Analizando..."
                    for model_name in ['XGBoost', 'RandomForest', 'SVM']:
                        if model_name in self.current_prediction:
                            pred_data = self.current_prediction[model_name]
                            if 'most_common' in pred_data and pred_data['most_common']:
                                activity, count = pred_data['most_common']
                                percentage = (count / pred_data['total_frames']) * 100
                                pred_text = f"{activity} ({percentage:.0f}%)"
                                break
                    
                    # Dibujar texto en el frame
                    cv2.putText(frame, pred_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Mostrar información del buffer
                buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}"
                cv2.putText(frame, buffer_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Mostrar el frame en una ventana
            cv2.imshow(window_name, frame)
            
            # Salir si se presiona 'q' o se cierra la ventana
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.root.after(0, self.stop_webcam)
                break
            
            # Verificar si la ventana fue cerrada
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.root.after(0, self.stop_webcam)
                break
            
            # Pequeña pausa para no saturar el CPU
            time.sleep(0.033)  # ~30 FPS
        
        # Cerrar ventana al salir
        cv2.destroyWindow(window_name)
        
        # Limpiar al salir
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def process_realtime_buffer(self):
        """Procesar el buffer de frames y hacer predicciones"""
        if len(self.frame_buffer) < self.buffer_size:
            return
        
        try:
            # Usar solo los últimos buffer_size frames (última ventana completa)
            buffer_to_process = self.frame_buffer[-self.buffer_size:].copy()
            
            # Preprocesar datos del buffer
            processed_data = self.preprocess_data(buffer_to_process)
            
            if processed_data is None or len(processed_data) == 0:
                return
            
            # Para tiempo real, usar solo la última ventana (última fila)
            if len(processed_data) > 1:
                processed_data = processed_data.iloc[[-1]]  # Última ventana
            
            # Hacer predicciones con todos los modelos
            predictions = self.predict_all_models(processed_data)
            
            # Guardar predicción actual para mostrar en el video
            self.current_prediction = predictions
            
            # Actualizar UI con resultados
            self.root.after(0, lambda: self.update_realtime_results(predictions))
            
        except Exception as e:
            # Mostrar error en la UI
            import traceback
            error_msg = f"Error al procesar: {str(e)}"
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"\n{error_msg}\n"))
    
    def update_realtime_results(self, predictions):
        """Actualizar resultados en tiempo real en la UI"""
        # Limpiar resultados anteriores pero mantener encabezado
        self.results_text.delete(1.0, tk.END)
        
        # Encabezado
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, "ANÁLISIS EN TIEMPO REAL - WEBCAM\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")
        
        # Información de estado
        self.results_text.insert(tk.END, f"Frames capturados: {self.frame_count}\n")
        self.results_text.insert(tk.END, f"Buffer: {len(self.frame_buffer)}/{self.buffer_size} frames\n")
        self.results_text.insert(tk.END, "-" * 70 + "\n\n")
        
        # Resultados de cada modelo
        self.results_text.insert(tk.END, "PREDICCIONES ACTUALES:\n")
        self.results_text.insert(tk.END, "-" * 70 + "\n")
        
        model_results = []
        for model_name, pred_data in predictions.items():
            if 'error' in pred_data:
                error_short = pred_data['error'].split('\n')[0][:50]
                model_results.append(f"{model_name:15s}: ERROR - {error_short}...")
                continue
            
            if pred_data['most_common']:
                activity, count = pred_data['most_common']
                percentage = (count / pred_data['total_frames']) * 100
                model_results.append(f"{model_name:15s}: {activity:20s} ({percentage:5.1f}%)")
        
        # Ordenar resultados para mostrar XGBoost primero (generalmente más preciso)
        priority_order = ['XGBoost', 'RandomForest', 'SVM']
        sorted_results = []
        for priority in priority_order:
            for result in model_results:
                if result.startswith(priority):
                    sorted_results.append(result)
                    model_results.remove(result)
                    break
        sorted_results.extend(model_results)  # Agregar los restantes
        
        for result in sorted_results:
            self.results_text.insert(tk.END, result + "\n")
        
        # Distribución completa (si hay múltiples actividades detectadas)
        self.results_text.insert(tk.END, "\n" + "-" * 70 + "\n")
        self.results_text.insert(tk.END, "DISTRIBUCIÓN DETALLADA:\n")
        self.results_text.insert(tk.END, "-" * 70 + "\n")
        
        # Mostrar distribución del mejor modelo (XGBoost si está disponible)
        best_model = None
        for model_name in ['XGBoost', 'RandomForest', 'SVM']:
            if model_name in predictions and 'frequencies' in predictions[model_name]:
                best_model = predictions[model_name]
                break
        
        if best_model and 'frequencies' in best_model:
            sorted_freq = sorted(best_model['frequencies'].items(), key=lambda x: x[1], reverse=True)
            for activity, count in sorted_freq:
                percentage = (count / best_model['total_frames']) * 100
                bar_length = int(percentage / 2)  # Barra visual
                bar = "█" * bar_length
                self.results_text.insert(tk.END, f"  {activity:20s}: {percentage:5.1f}% {bar}\n")
        
        # Timestamp
        self.results_text.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.results_text.insert(tk.END, f"Última actualización: {pd.Timestamp.now().strftime('%H:%M:%S')}\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        
        # Scroll al final
        self.results_text.see(tk.END)
    
    def on_closing(self):
        """Manejar cierre de la aplicación"""
        if self.webcam_active:
            self.stop_webcam()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PoseAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

