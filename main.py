import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageStat
import threading
import time
import os
import sys
import colorsys
import math
from collections import deque
from statistics import mode, mean
import pandas as pd

# ==========================================
# 0. SETTINGS & VARIABLES
# ==========================================
AGE_RANGE_OFFSET = 3
CLOTHING_THRESHOLD = 0.6

# --- LIBRARIES ---
try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️ ONNX Runtime missing. Age/Race won't work.")

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️ TensorFlow missing. Gender/Hair won't work.")

try:
    import torch
    import torchvision.transforms as T
    from transformers import ViTForImageClassification, ViTConfig, AutoImageProcessor, AutoModelForObjectDetection
    from safetensors.torch import load_file

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch/Transformers missing. Skin/Clothing won't work.")

try:
    import mediapipe as mp

    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ MediaPipe missing. Pose-based clothing detection won't work.")

try:
    from sklearn.neighbors import KNeighborsClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ Sklearn missing. Advanced color detection won't work.")

DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"


# ==========================================
# 1. ADVANCED COLOR DETECTOR (CSV-based AI)
# ==========================================
class AdvancedColorDetector:
    def __init__(self, csv_path='color_names.csv'):
        self.model = None
        if HAS_SKLEARN and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                raw_rgb = self.df[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']].values
                X = self._preprocess(raw_rgb)
                y = self.df['Name']

                self.model = KNeighborsClassifier(n_neighbors=1)
                self.model.fit(X, y)
                print("✅ Advanced Color AI Loaded")
            except Exception as e:
                print(f"⚠️ Color CSV loading failed: {e}")
                self.model = None

    def _preprocess(self, rgb_values):
        """Lighting-invariant HSV feature extraction"""
        features = []
        if len(rgb_values.shape) == 1:
            rgb_values = rgb_values.reshape(1, -1)

        for rgb in rgb_values:
            r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h_rad = h * 2 * math.pi
            features.append([math.sin(h_rad), math.cos(h_rad), s, v * 0.5])

        return np.array(features)

    def get_color_name(self, pil_crop):
        """Get color name from PIL crop using AI model"""
        if pil_crop.width < 1 or pil_crop.height < 1:
            return "Unknown", (128, 128, 128)

        try:
            # Get average color using ImageStat
            stat = ImageStat.Stat(pil_crop)
            avg_rgb = stat.mean[:3]
            r, g, b = int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])

            if self.model:
                # Use AI prediction
                input_rgb = np.array([[r, g, b]])
                processed_input = self._preprocess(input_rgb)
                prediction = self.model.predict(processed_input)
                return prediction[0], (r, g, b)
            else:
                # Fallback to basic detection
                return self._fallback_color(r, g, b), (r, g, b)
        except:
            return "Unknown", (128, 128, 128)

    def _fallback_color(self, r, g, b):
        """Simple fallback if AI model not available"""
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        H, S, V = h * 360, s * 100, v * 100

        if V < 20: return "Black"
        if S < 15: return "White" if V > 80 else "Grey"

        if H < 30:
            return "Red"
        elif H < 60:
            return "Orange"
        elif H < 90:
            return "Yellow"
        elif H < 150:
            return "Green"
        elif H < 210:
            return "Blue"
        elif H < 270:
            return "Purple"
        elif H < 330:
            return "Pink"
        else:
            return "Red"


# ==========================================
# 2. CLOTHING DETECTOR (Hybrid: Object Detection + Pose Colors)
# ==========================================
class ClothingDetector:
    def __init__(self):
        self.pose = None
        self.color_detector = AdvancedColorDetector()
        self.use_mediapipe = False

        # Object detection models
        self.model_cloth = None
        self.proc_cloth = None
        self.has_object_detection = False

        # Initialize MediaPipe
        if HAS_MEDIAPIPE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    min_detection_confidence=0.5,
                    model_complexity=1
                )
                self.use_mediapipe = True
                print("✅ MediaPipe Pose Loaded")
            except Exception as e:
                print(f"⚠️ MediaPipe failed: {e}")
                print("   → Will use object detection boxes for colors")
                self.pose = None
                self.use_mediapipe = False

    def load_object_detection(self):
        """Load fashion object detection model"""
        if not HAS_TORCH:
            return False

        try:
            path = "./fashion_model" if os.path.exists("./fashion_model") else "yainage90/fashion-object-detection"
            self.proc_cloth = AutoImageProcessor.from_pretrained(path)
            self.model_cloth = AutoModelForObjectDetection.from_pretrained(path).to(DEVICE)
            self.has_object_detection = True
            print("✅ Fashion Object Detection Loaded")
            return True
        except Exception as e:
            print(f"⚠️ Fashion model failed: {e}")
            return False

    def detect_clothing(self, cv2_frame, face_box=None):
        """Hybrid detection: Object detection for garment type + Pose for color"""

        if not self.has_object_detection:
            # Fallback to simple region detection
            return self._detect_with_regions(cv2_frame, face_box)

        try:
            # Step 1: Detect garments using object detection
            pil_frame = Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB))
            inputs = self.proc_cloth(images=pil_frame, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = self.model_cloth(**inputs)

            target = torch.tensor([pil_frame.size[::-1]])
            results = self.proc_cloth.post_process_object_detection(
                outputs, target_sizes=target, threshold=CLOTHING_THRESHOLD
            )[0]

            if len(results["scores"]) == 0:
                # No garments detected, try fallback
                return self._detect_with_regions(cv2_frame, face_box)

            # Step 2: Get pose landmarks for color sampling
            landmarks = None
            if self.use_mediapipe and self.pose:
                try:
                    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(rgb_frame)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                except:
                    pass

            # Step 3: Process each detected garment
            items = []
            h, w = cv2_frame.shape[:2]

            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                label = self.model_cloth.config.id2label[label_id.item()]
                box_coords = box.cpu().numpy().astype(int)
                x1, y1, x2, y2 = box_coords

                # Boundary check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Get color using pose-guided sampling
                color_name, rgb_val = self._get_garment_color(
                    pil_frame, label, box_coords, landmarks, w, h
                )

                items.append((color_name, label, rgb_val, [x1, y1, x2, y2]))

            return items

        except Exception as e:
            print(f"⚠️ Clothing detection error: {e}")
            return self._detect_with_regions(cv2_frame, face_box)

    def _get_garment_color(self, pil_frame, label, box, landmarks, width, height):
        """Get color using pose-guided sampling or box center"""

        # Try pose-guided sampling first
        if landmarks:
            sample_point = self._get_pose_sample_point(label, landmarks, width, height)
            if sample_point:
                px, py = sample_point
                x1, y1, x2, y2 = box

                # Check if point is inside or near the box
                if x1 - 20 < px < x2 + 20 and y1 - 20 < py < y2 + 20:
                    # Sample small region around point
                    px = max(0, min(width - 1, px))
                    py = max(0, min(height - 1, py))

                    r = 15
                    crop_x1 = max(0, px - r)
                    crop_y1 = max(0, py - r)
                    crop_x2 = min(width, px + r)
                    crop_y2 = min(height, py + r)

                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        crop = pil_frame.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        return self.color_detector.get_color_name(crop)

        # Fallback: Sample from center of detection box
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # Sample from center 30% of box
        cx1 = int(x1 + bw * 0.35)
        cy1 = int(y1 + bh * 0.35)
        cx2 = int(x2 - bw * 0.35)
        cy2 = int(y2 - bh * 0.35)

        if cx2 > cx1 and cy2 > cy1:
            crop = pil_frame.crop((cx1, cy1, cx2, cy2))
            return self.color_detector.get_color_name(crop)

        return "Unknown", (128, 128, 128)

    def _get_pose_sample_point(self, label, landmarks, width, height):
        """Get optimal sampling point based on garment type and pose"""
        if not landmarks:
            return None

        label_lower = label.lower()

        def get_xy(idx):
            return int(landmarks[idx].x * width), int(landmarks[idx].y * height)

        # SHOES/FOOTWEAR
        if any(x in label_lower for x in ["shoe", "boot", "sneaker", "sandal"]):
            if landmarks[31].visibility > landmarks[32].visibility:
                return get_xy(31)
            else:
                return get_xy(32)

        # UPPER BODY (shirts, jackets, coats, sweaters, tops, dresses)
        if any(x in label_lower for x in
               ["shirt", "jacket", "coat", "sweater", "top", "dress", "blouse", "cardigan", "hoodie", "vest"]):
            # Sample below shoulder
            if landmarks[11].visibility > landmarks[12].visibility:
                x, y = get_xy(11)
                return (x, y + 20)
            else:
                x, y = get_xy(12)
                return (x, y + 20)

        # LOWER BODY (pants, shorts, skirts)
        if any(x in label_lower for x in ["pant", "trouser", "jean", "short", "skirt", "legging"]):
            # Sample between hip and knee
            if landmarks[23].visibility > landmarks[24].visibility:
                h_x, h_y = get_xy(23)
                k_x, k_y = get_xy(25)
                return ((h_x + k_x) // 2, (h_y + k_y) // 2)
            else:
                h_x, h_y = get_xy(24)
                k_x, k_y = get_xy(26)
                return ((h_x + k_x) // 2, (h_y + k_y) // 2)

        return None

    def _detect_with_regions(self, cv2_frame, face_box):
        """Fallback: Simple region-based detection relative to face"""
        if face_box is None:
            return []

        h, w = cv2_frame.shape[:2]
        fx, fy, fw, fh = face_box

        face_center_x = fx + fw // 2
        face_bottom = fy + fh

        items = []
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Upper Body
        upper_y = face_bottom + 20
        upper_x = face_center_x - 30
        if 0 < upper_y < h - 60 and 0 < upper_x < w - 60:
            x1, y1 = max(0, upper_x), max(0, upper_y)
            x2, y2 = min(w, upper_x + 60), min(h, upper_y + 60)
            crop = pil_frame.crop((x1, y1, x2, y2))
            color_name, rgb_val = self.color_detector.get_color_name(crop)
            items.append((color_name, "Upper Body", rgb_val, [x1, y1, x2, y2]))

        # Lower Body
        lower_y = face_bottom + 120
        lower_x = face_center_x - 30
        if 0 < lower_y < h - 60 and 0 < lower_x < w - 60:
            x1, y1 = max(0, lower_x), max(0, lower_y)
            x2, y2 = min(w, lower_x + 60), min(h, lower_y + 60)
            crop = pil_frame.crop((x1, y1, x2, y2))
            color_name, rgb_val = self.color_detector.get_color_name(crop)
            items.append((color_name, "Lower Body", rgb_val, [x1, y1, x2, y2]))

        return items


# ==========================================
# 3. RESULT STABILIZER
# ==========================================
class ResultStabilizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = {
            "age": [], "race": [], "gender": [], "tone": [], "hair": []
        }

    def update(self, category, value):
        if value not in ["N/A", "Err", "--"]:
            self.history[category].append(value)

    def get_final_result(self, category):
        buffer = self.history[category]
        if not buffer: return "--"

        if category == "age":
            try:
                avg = mean([int(x) for x in buffer])
                return str(int(avg))
            except:
                return buffer[-1]
        else:
            try:
                return mode(buffer)
            except:
                return buffer[-1]


# ==========================================
# 4. MODEL MANAGER
# ==========================================
class ModelManager:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.clothing_detector = ClothingDetector()

        self.status = {
            "gender": False, "age": False, "race": False, "hair": False,
            "tone": False, "clothing": False
        }

        # Race labels for new model
        self.race_labels = [
            'Black', 'East Asian', 'Indian', 'Latino_Hispanic',
            'Middle Eastern', 'Southeast Asian', 'White'
        ]

        self.tone_labels = [
            "Type 1 (Pale)", "Type 2 (Fair)", "Type 3 (Medium)",
            "Type 4 (Olive)", "Type 5 (Brown)", "Type 6 (Dark)"
        ]

        self.sess_age = None
        self.sess_race = None
        self.model_gender = None
        self.model_hair = None
        self.model_tone = None

    def load_models(self):
        print("--- Loading Models ---")

        # Separate Age Model
        if HAS_ONNX and os.path.exists("age_race.onnx"):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.sess_age = ort.InferenceSession("age_race.onnx", providers=providers)
                self.input_name_age = self.sess_age.get_inputs()[0].name
                self.status["age"] = True
                print("✅ Age Model Loaded")
            except Exception as e:
                print(f"⚠️ Age model failed: {e}")

        # New Separate Race Model
        if HAS_ONNX and os.path.exists("race_model.onnx"):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.sess_race = ort.InferenceSession("race_model.onnx", providers=providers)
                self.input_name_race = self.sess_race.get_inputs()[0].name
                self.status["race"] = True
                print("✅ Race Model Loaded")
            except Exception as e:
                print(f"⚠️ Race model failed: {e}")

        # Gender Model
        if HAS_TF and os.path.exists("gender_recognition_xception_finetuned.keras"):
            try:
                self.model_gender = tf.keras.models.load_model(
                    "gender_recognition_xception_finetuned.keras", compile=False
                )
                self.status["gender"] = True
                print("✅ Gender Loaded")
            except:
                pass

        # Hair Model
        if HAS_TF and os.path.exists("hair_segmentation_model.h5"):
            try:
                self.model_hair = tf.keras.models.load_model("hair_segmentation_model.h5", compile=False)
                self.status["hair"] = True
                print("✅ Hair Loaded")
            except:
                pass

        # Skin Tone Model
        if HAS_TORCH and os.path.exists("skin_tone_model.safetensors"):
            try:
                from huggingface_hub import snapshot_download
                if not os.path.exists("config.json"):
                    snapshot_download(
                        repo_id="google/vit-large-patch16-224-in21k",
                        allow_patterns=["config.json"],
                        local_dir="."
                    )

                config = ViTConfig.from_pretrained(".", num_labels=6)
                self.model_tone = ViTForImageClassification(config)
                state_dict = load_file("skin_tone_model.safetensors")
                new_state = {}
                for k, v in state_dict.items():
                    nk = k.replace("model.", "vit.") if "model." in k else k
                    new_state[nk] = v
                self.model_tone.load_state_dict(new_state, strict=False)
                self.model_tone.to(DEVICE).eval()
                self.status["tone"] = True
                print("✅ Skin Tone Loaded")
            except:
                pass

        # Load clothing detection
        if self.clothing_detector.load_object_detection():
            self.status["clothing"] = True
            print("✅ Clothing Detection Ready (Hybrid Mode)")
        else:
            self.status["clothing"] = True
            print("✅ Clothing Detection Ready (Fallback Mode)")

    # --- PREDICTIONS ---
    def predict_age(self, crop):
        """Use age_race.onnx but only for age"""
        if not self.status["age"]:
            return "N/A"
        try:
            img = cv2.resize(crop, (224, 224)).astype(np.float32)
            blob = np.expand_dims(img, axis=0)
            outputs = self.sess_age.run(None, {self.input_name_age: blob})
            age = int(outputs[0][0][0])  # First output is age
            return f"{age}"
        except:
            return "Err"

    def predict_race(self, crop):
        """Use new race_model.onnx with Black/Indian detection"""
        if not self.status["race"]:
            return "N/A"
        try:
            # Preprocess for race model (normalize to 0-1 like test code)
            img = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
            blob = np.expand_dims(img, axis=0)

            # Run inference
            outputs = self.sess_race.run(None, {self.input_name_race: blob})
            probs = outputs[0][0]  # Get probability array

            # Filter out unreliable predictions
            latino_idx = self.race_labels.index('Latino_Hispanic')
            east_asian_idx = self.race_labels.index('East Asian')
            se_asian_idx = self.race_labels.index('Southeast Asian')  # <--- PHANTOM INDEX ADDED

            # Suppress these categories by zeroing their probabilities
            probs[latino_idx] = 0.0
            probs[east_asian_idx] = 0.0
            probs[se_asian_idx] = 0.0  # <--- PHANTOM SUPPRESSION ADDED

            # Get new prediction after filtering
            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx] * 100  # Convert to percentage

            # Apply Black/Indian disambiguation logic (73% threshold from test code)
            label = self.race_labels[predicted_idx]

            if label == "Black" and confidence < 73.0:
                return "Indian"

            return label
        except Exception as e:
            print(f"Race prediction error: {e}")
            return "Err"

    def predict_gender(self, crop):
        if not self.status["gender"]:
            return "N/A"
        try:
            img = cv2.resize(crop, (299, 299))
            blob = np.expand_dims(img, axis=0)
            pred = self.model_gender.predict(blob, verbose=0)[0][0]
            return "Male" if pred > 0.5 else "Female"
        except:
            return "Err"

    def predict_hair(self, crop):
        if not self.status["hair"]:
            return "N/A"
        try:
            img = cv2.resize(crop, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            blob = np.expand_dims(img, axis=0)
            pred = self.model_hair.predict(blob, verbose=0)
            mask = np.argmax(pred, axis=-1)[0]
            if np.sum(mask == 2) < 200:
                return "Bald/No Hair"
            ys, xs = np.where(mask == 2)
            if np.max(ys) > 180:
                return "Long Hair"
            return "Short Hair"
        except:
            return "Err"

    def predict_tone(self, crop):
        if not self.status["tone"]:
            return "N/A"
        try:
            img_f = crop.astype(np.float32)
            avg = np.mean(img_f, axis=(0, 1))
            g_mean = np.mean(avg)
            scale = g_mean / (avg + 1e-6)
            img_fix = np.clip(img_f * scale, 0, 255).astype(np.uint8)

            pil = Image.fromarray(cv2.cvtColor(img_fix, cv2.COLOR_BGR2RGB))
            t_ops = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3)
            ])
            t_in = t_ops(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.model_tone(t_in)
            return self.tone_labels[torch.argmax(out.logits, 1).item()]
        except:
            return "Err"

    def predict_clothing(self, full_frame, face_box=None):
        """Detect clothing with hybrid approach"""
        if not self.status["clothing"]:
            return []
        return self.clothing_detector.detect_clothing(full_frame, face_box)


# ==========================================
# 5. GUI APP
# ==========================================
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Capstone Profiler")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")

        self.manager = ModelManager()
        self.stabilizer = ResultStabilizer()

        self.state = "IDLE"
        self.countdown_val = 0
        self.scan_start_time = 0
        self.cap = None

        self.setup_ui()
        threading.Thread(target=self.load_models_thread, daemon=True).start()

    def setup_ui(self):
        head = tk.Frame(self.root, bg="#222", height=60, pady=10)
        head.pack(fill=tk.X)

        tk.Label(
            head, text="AI PROFILER v2.0", bg="#222",
            fg="#00ffcc", font=("Helvetica", 18, "bold")
        ).pack(side=tk.LEFT, padx=20)

        self.status_lbl = tk.Label(head, text="Loading...", bg="#222", fg="orange")
        self.status_lbl.pack(side=tk.LEFT, padx=20)

        self.btn_action = tk.Button(
            head, text="START CAMERA", command=self.toggle_cam,
            bg="#444", fg="white", font=("Arial", 12, "bold"), width=20
        )
        self.btn_action.pack(side=tk.RIGHT, padx=20)

        body = tk.Frame(self.root, bg="#1a1a1a")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cam_lbl = tk.Label(body, bg="black", text="[Camera Off]", fg="#555")
        self.cam_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(body, bg="#2b2b2b", width=400)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        panel.pack_propagate(False)

        def mk_card(title, key):
            f = tk.Frame(panel, bg="#333", pady=8, padx=10)
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=title, bg="#333", fg="#aaa", font=("Arial", 9)).pack(anchor="w")
            l = tk.Label(f, text="--", bg="#333", fg="white", font=("Arial", 16, "bold"))
            l.pack(anchor="w")
            setattr(self, f"lbl_{key}", l)

        mk_card("AGE", "age")
        mk_card("GENDER", "gender")
        mk_card("ETHNICITY", "race")
        mk_card("SKIN TONE", "tone")
        mk_card("HAIR", "hair")

        tk.Label(panel, text="CLOTHING", bg="#2b2b2b", fg="#aaa").pack(anchor="w", padx=10, pady=(20, 5))

        # Custom clothing frame with canvas for colored boxes
        cloth_frame = tk.Frame(panel, bg="#222")
        cloth_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Scrollable canvas for clothing items
        self.cloth_canvas = tk.Canvas(cloth_frame, bg="#222", highlightthickness=0)
        scrollbar = tk.Scrollbar(cloth_frame, orient="vertical", command=self.cloth_canvas.yview)
        self.cloth_scrollable = tk.Frame(self.cloth_canvas, bg="#222")

        self.cloth_scrollable.bind(
            "<Configure>",
            lambda e: self.cloth_canvas.configure(scrollregion=self.cloth_canvas.bbox("all"))
        )

        self.cloth_canvas.create_window((0, 0), window=self.cloth_scrollable, anchor="nw")
        self.cloth_canvas.configure(yscrollcommand=scrollbar.set)

        self.cloth_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def load_models_thread(self):
        self.manager.load_models()
        self.root.after(0, lambda: self.status_lbl.config(text="Models Ready", fg="#00ff00"))

    def toggle_cam(self):
        if self.state == "IDLE":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return

            self.state = "PREVIEW"
            self.btn_action.config(text="SCAN SUBJECT", bg="#0055ff")
            threading.Thread(target=self.loop, daemon=True).start()

        elif self.state == "PREVIEW":
            self.state = "COUNTDOWN"
            self.countdown_val = 3
            self.stabilizer.reset()
            self.btn_action.config(text="SCANNING...", state=tk.DISABLED, bg="#cc0000")
            self.update_labels(clear=True)
            self.root.after(1000, self.countdown_tick)

        elif self.state == "DONE":
            self.state = "PREVIEW"
            self.btn_action.config(text="SCAN SUBJECT", bg="#0055ff")

    def countdown_tick(self):
        if self.countdown_val > 1:
            self.countdown_val -= 1
            self.root.after(1000, self.countdown_tick)
        else:
            self.state = "SCANNING"
            self.scan_start_time = time.time()

    def loop(self):
        while True:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            display = frame.copy()
            h, w = frame.shape[:2]

            if self.state == "PREVIEW":
                cv2.putText(display, "READY - CLICK SCAN", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.draw_preview_face(display, frame)

            elif self.state == "COUNTDOWN":
                text = str(self.countdown_val)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
                cx, cy = w // 2 - tw // 2, h // 2 + th // 2
                cv2.putText(display, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

            elif self.state == "SCANNING":
                cv2.putText(display, "SCANNING... STAY STILL", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.analyze_frame(frame)
                if time.time() - self.scan_start_time > 2.5:
                    self.finish_scan()

            elif self.state == "DONE":
                cv2.putText(display, "COMPLETE", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            cw = self.cam_lbl.winfo_width()
            ch = self.cam_lbl.winfo_height()
            if cw > 10:
                pil.thumbnail((cw, ch))
            imgtk = ImageTk.PhotoImage(pil)
            self.root.after(0, lambda i=imgtk: self.update_cam_lbl(i))

            time.sleep(0.03)

    def draw_preview_face(self, display, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.manager.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

    def analyze_frame(self, frame):
        # Face Analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.manager.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        face_box = None
        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]
            face_box = (x, y, w, h)

            H, W = frame.shape[:2]
            p = 20
            y1, x1 = max(0, y - p), max(0, x - p)
            y2, x2 = min(H, y + h + p), min(W, x + w + p)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                a = self.manager.predict_age(crop)
                r = self.manager.predict_race(crop)
                g = self.manager.predict_gender(crop)
                t = self.manager.predict_tone(crop)
                h_style = self.manager.predict_hair(crop)

                self.stabilizer.update("age", a)
                self.stabilizer.update("race", r)
                self.stabilizer.update("gender", g)
                self.stabilizer.update("tone", t)
                self.stabilizer.update("hair", h_style)

        # Clothing Detection (Pass face_box for fallback positioning)
        self.last_clothes = self.manager.predict_clothing(frame, face_box)

    def finish_scan(self):
        self.state = "DONE"
        self.root.after(0, self.show_final_results)

    def show_final_results(self):
        self.btn_action.config(text="RESET", state=tk.NORMAL, bg="#444")
        self.update_labels()

    def update_cam_lbl(self, img):
        self.cam_lbl.configure(image=img)
        self.cam_lbl.image = img

    def update_labels(self, clear=False):
        if clear:
            # Clear clothing display
            for widget in self.cloth_scrollable.winfo_children():
                widget.destroy()
        else:
            # Age Range
            stable_age_str = self.stabilizer.get_final_result("age")
            if stable_age_str.isdigit():
                age_val = int(stable_age_str)
                age_text = f"{age_val - AGE_RANGE_OFFSET} - {age_val + AGE_RANGE_OFFSET}"
            else:
                age_text = stable_age_str

            self.lbl_age.config(text=age_text)
            self.lbl_race.config(text=self.stabilizer.get_final_result("race"))
            self.lbl_gender.config(text=self.stabilizer.get_final_result("gender"))
            self.lbl_tone.config(text=self.stabilizer.get_final_result("tone"))
            self.lbl_hair.config(text=self.stabilizer.get_final_result("hair"))

            # Clear and update clothing items with visual color boxes
            for widget in self.cloth_scrollable.winfo_children():
                widget.destroy()

            if hasattr(self, 'last_clothes'):
                for item in self.last_clothes:
                    color_name, garment, rgb_val, box = item

                    # Create item frame
                    item_frame = tk.Frame(self.cloth_scrollable, bg="#222", pady=5)
                    item_frame.pack(fill=tk.X, padx=5, pady=2)

                    # Color swatch (small box)
                    hex_color = f'#{rgb_val[0]:02x}{rgb_val[1]:02x}{rgb_val[2]:02x}'
                    color_box = tk.Label(
                        item_frame,
                        bg=hex_color,
                        width=3,
                        height=1,
                        relief=tk.SOLID,
                        borderwidth=1
                    )
                    color_box.pack(side=tk.LEFT, padx=(0, 8))

                    # Text label with color name and RGB
                    text_label = tk.Label(
                        item_frame,
                        text=f"{color_name} {garment}",
                        bg="#222",
                        fg="#00ffcc",
                        font=("Arial", 11, "bold"),
                        anchor="w"
                    )
                    text_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

                    # RGB values (faint)
                    rgb_label = tk.Label(
                        item_frame,
                        text=f"RGB({rgb_val[0]}, {rgb_val[1]}, {rgb_val[2]})",
                        bg="#222",
                        fg="#555",
                        font=("Courier", 8),
                        anchor="e"
                    )
                    rgb_label.pack(side=tk.RIGHT, padx=(5, 0))


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()