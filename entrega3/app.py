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
import math

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
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
        
        # Crear interfaz primero
        self.create_ui()
        
        # Cargar modelos después de crear la UI
        self.load_models()
        
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
            # Intentar obtener desde XGBoost primero (es el más estricto con el orden)
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
            
            # Si aún no se tiene, usar el orden del error (hardcoded como último recurso)
            if self.expected_feature_order is None:
                # Orden exacto según el error de XGBoost
                self.expected_feature_order = [
                    'x_0', 'x_1', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 
                    'x_2', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 
                    'x_3', 'x_30', 'x_31', 'x_32', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9',
                    'y_0', 'y_1', 'y_10', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 
                    'y_2', 'y_20', 'y_21', 'y_22', 'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 
                    'y_3', 'y_30', 'y_31', 'y_32', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9',
                    'z_0', 'z_1', 'z_10', 'z_11', 'z_12', 'z_13', 'z_14', 'z_15', 'z_16', 'z_17', 'z_18', 'z_19', 
                    'z_2', 'z_20', 'z_21', 'z_22', 'z_23', 'z_24', 'z_25', 'z_26', 'z_27', 'z_28', 'z_29', 
                    'z_3', 'z_30', 'z_31', 'z_32', 'z_4', 'z_5', 'z_6', 'z_7', 'z_8', 'z_9',
                    'frame_id',
                    'vel_x_0', 'vel_x_1', 'vel_x_10', 'vel_x_11', 'vel_x_12', 'vel_x_13', 'vel_x_14', 'vel_x_15', 'vel_x_16', 'vel_x_17', 'vel_x_18', 'vel_x_19', 
                    'vel_x_2', 'vel_x_20', 'vel_x_21', 'vel_x_22', 'vel_x_23', 'vel_x_24', 'vel_x_25', 'vel_x_26', 'vel_x_27', 'vel_x_28', 'vel_x_29', 
                    'vel_x_3', 'vel_x_30', 'vel_x_31', 'vel_x_32', 'vel_x_4', 'vel_x_5', 'vel_x_6', 'vel_x_7', 'vel_x_8', 'vel_x_9',
                    'vel_y_0', 'vel_y_1', 'vel_y_10', 'vel_y_11', 'vel_y_12', 'vel_y_13', 'vel_y_14', 'vel_y_15', 'vel_y_16', 'vel_y_17', 'vel_y_18', 'vel_y_19', 
                    'vel_y_2', 'vel_y_20', 'vel_y_21', 'vel_y_22', 'vel_y_23', 'vel_y_24', 'vel_y_25', 'vel_y_26', 'vel_y_27', 'vel_y_28', 'vel_y_29', 
                    'vel_y_3', 'vel_y_30', 'vel_y_31', 'vel_y_32', 'vel_y_4', 'vel_y_5', 'vel_y_6', 'vel_y_7', 'vel_y_8', 'vel_y_9',
                    'vel_z_0', 'vel_z_1', 'vel_z_10', 'vel_z_11', 'vel_z_12', 'vel_z_13', 'vel_z_14', 'vel_z_15', 'vel_z_16', 'vel_z_17', 'vel_z_18', 'vel_z_19', 
                    'vel_z_2', 'vel_z_20', 'vel_z_21', 'vel_z_22', 'vel_z_23', 'vel_z_24', 'vel_z_25', 'vel_z_26', 'vel_z_27', 'vel_z_28', 'vel_z_29', 
                    'vel_z_3', 'vel_z_30', 'vel_z_31', 'vel_z_32', 'vel_z_4', 'vel_z_5', 'vel_z_6', 'vel_z_7', 'vel_z_8', 'vel_z_9',
                    'angle_brazo_derecho', 'inclinacion_tronco'
                ]
            
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
        
        # Label para mostrar video seleccionado
        self.video_label = ttk.Label(main_frame, text="Ningún video seleccionado", foreground="gray")
        self.video_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Status label (usar tk.Label para soportar fg)
        self.status_label = tk.Label(main_frame, text="Listo", fg="blue")
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Frame para resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de los Modelos", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
        main_frame.rowconfigure(5, weight=1)
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
        """Extraer landmarks del video usando MediaPipe (igual que MediaPipe.py)"""
        landmarks_list = []
        
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
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_data = {}
                    
                    # Extraer landmarks exactamente como en MediaPipe.py
                    # explicitly iterate NUM_LANDMARKS to keep column schema stable
                    for i in range(NUM_LANDMARKS):
                        lm = results.pose_landmarks.landmark[i]
                        frame_data[f'x_{i}'] = lm.x
                        frame_data[f'y_{i}'] = lm.y
                        frame_data[f'z_{i}'] = lm.z
                        frame_data[f'visibility_{i}'] = lm.visibility
                    
                    landmarks_list.append(frame_data)
                
                frame_count += 1
                
                # Actualizar progreso cada 10 frames
                if frame_count % 10 == 0:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.root.after(0, lambda p=progress: self.status_label.config(
                        text=f"Extrayendo landmarks... {p:.1f}%", fg="blue"))
            
            cap.release()
        
        return landmarks_list
    
    def preprocess_data(self, landmarks_data):
        """Aplicar el mismo preprocesamiento que en el notebook"""
        if not landmarks_data:
            return None
        
        df = pd.DataFrame(landmarks_data)
        
        columns_to_remove = [col for col in df.columns if 'visibility' in col]
        df = df.drop(columns=columns_to_remove)
        
        coord_cols = [f'{axis}_{i}' for i in range(NUM_LANDMARKS) for axis in ['x', 'y', 'z']]
        available_coords = [col for col in coord_cols if col in df.columns]
        
        if available_coords:
            df[available_coords] = self.coord_scaler.transform(df[available_coords])
        
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
        
        for i in range(NUM_LANDMARKS):
            for axis in ['x', 'y', 'z']:
                col = f'{axis}_{i}'
                if col in df.columns:
                    df[f'vel_{axis}_{i}'] = df[col].diff().fillna(0)
        
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
        
        try:
                df['inclinacion_tronco'] = np.degrees(np.arctan2(
                (df['y_24'] + df['y_23']) / 2 - (df['y_12'] + df['y_11']) / 2,
                (df['z_24'] + df['z_23']) / 2 - (df['z_12'] + df['z_11']) / 2
            ))
        except:
            df['inclinacion_tronco'] = 0
        
        df['frame_id'] = range(len(df))
        
        if hasattr(self, 'expected_feature_order') and self.expected_feature_order:
            expected_cols = self.expected_feature_order
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            
            return df[expected_cols]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            coord_cols = sorted([c for c in numeric_cols if any(c.startswith(f'{axis}_') for axis in ['x', 'y', 'z']) and not c.startswith('vel_') and c != 'frame_id'])
            frame_id_col = ['frame_id'] if 'frame_id' in df.columns else []
            vel_cols = sorted([c for c in numeric_cols if c.startswith('vel_')])
            derived_cols = []
            if 'angle_brazo_derecho' in df.columns:
                derived_cols.append('angle_brazo_derecho')
            if 'inclinacion_tronco' in df.columns:
                derived_cols.append('inclinacion_tronco')
            
            ordered_cols = coord_cols + frame_id_col + vel_cols + derived_cols
            
            available_cols = [col for col in ordered_cols if col in df.columns]
            
            return df[available_cols]
    
    def predict_all_models(self, processed_data):
        """Hacer predicciones con todos los modelos"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'SVM' and hasattr(model, 'named_steps'):
                    pass
                
                if hasattr(model, 'feature_names_in_'):
                    expected_cols = list(model.feature_names_in_)
                    missing_cols = [col for col in expected_cols if col not in processed_data.columns]
                    for col in missing_cols:
                        processed_data[col] = 0
                    processed_data = processed_data[expected_cols]
                elif model_name == 'XGBoost' and hasattr(model, 'get_booster'):
                    try:
                        expected_cols = model.get_booster().feature_names
                        if expected_cols:
                            missing_cols = [col for col in expected_cols if col not in processed_data.columns]
                            for col in missing_cols:
                                processed_data[col] = 0
                            processed_data = processed_data[expected_cols]
                    except:
                        pass
                
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
                predictions[model_name] = {'error': str(e)}
        
        return predictions
    
    def _show_results(self, predictions, landmarks_data):
        """Mostrar resultados en la interfaz"""
        self.results_text.delete(1.0, tk.END)
        
        # Encabezado
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, "RESULTADOS DEL ANÁLISIS DE POSES\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")
        
        self.results_text.insert(tk.END, f"Total de frames procesados: {len(landmarks_data)}\n\n")
        
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

def main():
    root = tk.Tk()
    app = PoseAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

