import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import colorsys
from collections import deque
from statistics import mode, mean

# ==========================================
# 0. SETTINGS & VARIABLES
# ==========================================
# Change this to widen/narrow the age range
# Example: If AI sees 25, and Offset is 3, it displays "22 - 28"
AGE_RANGE_OFFSET = 3

# Clothing Confidence (0.5 = 50% sure)
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

# --- CONFIGURATION ---
DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"


# ==========================================
# 1. COLOR LOGIC
# ==========================================
class GranularColorDetector:
    def get_color_name(self, pil_crop):
        if pil_crop.width < 1 or pil_crop.height < 1: return ""

        crop_small = pil_crop.resize((20, 20))
        data = np.array(crop_small)
        avg_rgb = np.mean(data.reshape(-1, 3), axis=0)
        r, g, b = avg_rgb

        h_norm, s_norm, v_norm = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        H, S, V = h_norm * 360, s_norm * 100, v_norm * 100

        if V < 20: return "Black"
        if S < 15:
            if V > 80:
                return "White"
            else:
                return "Grey"
        if 190 <= H < 240 and S < 30 and V > 85: return "White"

        adjective = ""
        if V < 40:
            adjective = "Dark "
        elif V > 90:
            adjective = "Bright "
        elif S < 40:
            adjective = "Pale "

        c = "Unknown"
        if (H >= 0 and H < 12) or (H >= 345 and H <= 360):
            c = "Red"
        elif 12 <= H < 35:
            c = "Beige" if V > 60 else "Brown"
        elif 35 <= H < 70:
            c = "Yellow"
        elif 70 <= H < 160:
            c = "Green"
        elif 160 <= H < 185:
            c = "Cyan"
        elif 185 <= H < 255:
            c = "Blue"
        elif 255 <= H < 295:
            c = "Purple"
        elif 295 <= H < 345:
            c = "Pink"

        return f"{adjective}{c}".strip()


# ==========================================
# 2. STABILIZER
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
                return str(int(avg))  # Returns single number string "25"
            except:
                return buffer[-1]
        else:
            try:
                return mode(buffer)
            except:
                return buffer[-1]


# ==========================================
# 3. MODEL MANAGER
# ==========================================
class ModelManager:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.color_detector = GranularColorDetector()

        self.status = {
            "gender": False, "age_race": False, "hair": False,
            "tone": False, "cloth": False
        }

        self.race_labels = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian',
                            'White']
        self.tone_labels = ["Type 1 (Pale)", "Type 2 (Fair)", "Type 3 (Medium)", "Type 4 (Olive)", "Type 5 (Brown)",
                            "Type 6 (Dark)"]

        self.sess_age_race = None
        self.model_gender = None
        self.model_hair = None
        self.model_tone = None
        self.model_cloth = None

    def load_models(self):
        print("--- Loading Models ---")

        if HAS_ONNX and os.path.exists("age_race.onnx"):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.sess_age_race = ort.InferenceSession("age_race.onnx", providers=providers)
                self.input_name_ar = self.sess_age_race.get_inputs()[0].name
                self.status["age_race"] = True
                print("✅ Age/Race Loaded")
            except:
                pass

        if HAS_TF and os.path.exists("gender_recognition_xception_finetuned.keras"):
            try:
                self.model_gender = tf.keras.models.load_model("gender_recognition_xception_finetuned.keras",
                                                               compile=False)
                self.status["gender"] = True
                print("✅ Gender Loaded")
            except:
                pass

        if HAS_TF and os.path.exists("hair_segmentation_model.h5"):
            try:
                self.model_hair = tf.keras.models.load_model("hair_segmentation_model.h5", compile=False)
                self.status["hair"] = True
                print("✅ Hair Loaded")
            except:
                pass

        if HAS_TORCH and os.path.exists("skin_tone_model.safetensors"):
            try:
                from huggingface_hub import snapshot_download
                if not os.path.exists("config.json"):
                    snapshot_download(repo_id="google/vit-large-patch16-224-in21k", allow_patterns=["config.json"],
                                      local_dir=".")

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

        if HAS_TORCH:
            try:
                path = "./fashion_model" if os.path.exists("./fashion_model") else "yainage90/fashion-object-detection"
                self.proc_cloth = AutoImageProcessor.from_pretrained(path)
                self.model_cloth = AutoModelForObjectDetection.from_pretrained(path).to(DEVICE)
                self.status["cloth"] = True
                print("✅ Clothing Loaded")
            except:
                pass

    # --- PREDICTIONS ---
    def predict_age_race(self, crop):
        if not self.status["age_race"]: return "N/A", "N/A"
        try:
            img = cv2.resize(crop, (224, 224)).astype(np.float32)
            blob = np.expand_dims(img, axis=0)
            outputs = self.sess_age_race.run(None, {self.input_name_ar: blob})
            age = int(outputs[0][0][0])
            race = self.race_labels[np.argmax(outputs[1][0])]
            return f"{age}", race
        except:
            return "Err", "Err"

    def predict_gender(self, crop):
        if not self.status["gender"]: return "N/A"
        try:
            img = cv2.resize(crop, (299, 299))
            blob = np.expand_dims(img, axis=0)
            pred = self.model_gender.predict(blob, verbose=0)[0][0]
            return "Male" if pred > 0.5 else "Female"
        except:
            return "Err"

    def predict_hair(self, crop):
        if not self.status["hair"]: return "N/A"
        try:
            img = cv2.resize(crop, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            blob = np.expand_dims(img, axis=0)
            pred = self.model_hair.predict(blob, verbose=0)
            mask = np.argmax(pred, axis=-1)[0]
            if np.sum(mask == 2) < 200: return "Bald/No Hair"
            ys, xs = np.where(mask == 2)
            if np.max(ys) > 180: return "Long Hair"
            return "Short Hair"
        except:
            return "Err"

    def predict_tone(self, crop):
        if not self.status["tone"]: return "N/A"
        try:
            img_f = crop.astype(np.float32)
            avg = np.mean(img_f, axis=(0, 1))
            g_mean = np.mean(avg)
            scale = g_mean / (avg + 1e-6)
            img_fix = np.clip(img_f * scale, 0, 255).astype(np.uint8)

            pil = Image.fromarray(cv2.cvtColor(img_fix, cv2.COLOR_BGR2RGB))
            t_ops = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
            t_in = t_ops(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.model_tone(t_in)
            return self.tone_labels[torch.argmax(out.logits, 1).item()]
        except:
            return "Err"

    def predict_cloth(self, full_img):
        if not self.status["cloth"]: return []
        try:
            pil = Image.fromarray(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))
            inputs = self.proc_cloth(images=pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.model_cloth(**inputs)

            target = torch.tensor([pil.size[::-1]])
            res = \
            self.proc_cloth.post_process_object_detection(outputs, target_sizes=target, threshold=CLOTHING_THRESHOLD)[0]

            items = []
            for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
                label = self.model_cloth.config.id2label[l.item()]
                box = b.cpu().numpy().astype(int)

                # Color Check
                x1, y1, x2, y2 = box
                h, w = full_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                item_crop = pil.crop((x1, y1, x2, y2))
                color = self.color_detector.get_color_name(item_crop)
                items.append((f"{color} {label}", box))
            return items
        except:
            return []


# ==========================================
# 4. GUI APP
# ==========================================
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capstone Profiler")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")

        self.manager = ModelManager()
        self.stabilizer = ResultStabilizer()

        # --- STATE MACHINE ---
        self.state = "IDLE"
        self.countdown_val = 0
        self.scan_start_time = 0
        self.cap = None

        self.setup_ui()
        threading.Thread(target=self.load_models_thread).start()

    def setup_ui(self):
        head = tk.Frame(self.root, bg="#222", height=60, pady=10)
        head.pack(fill=tk.X)

        tk.Label(head, text="AI PROFILER", bg="#222", fg="#00ffcc", font=("Helvetica", 18, "bold")).pack(side=tk.LEFT,
                                                                                                         padx=20)
        self.status_lbl = tk.Label(head, text="Loading...", bg="#222", fg="orange")
        self.status_lbl.pack(side=tk.LEFT, padx=20)

        # MAIN BUTTON
        self.btn_action = tk.Button(head, text="START CAMERA", command=self.toggle_cam,
                                    bg="#444", fg="white", font=("Arial", 12, "bold"), width=20)
        self.btn_action.pack(side=tk.RIGHT, padx=20)

        body = tk.Frame(self.root, bg="#1a1a1a")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera Area
        self.cam_lbl = tk.Label(body, bg="black", text="[Camera Off]", fg="#555")
        self.cam_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Result Panel
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
        self.cloth_list = tk.Listbox(panel, bg="#222", fg="#00ffcc", bd=0, height=12, font=("Arial", 12))
        self.cloth_list.pack(fill=tk.X, padx=10)

    def load_models_thread(self):
        self.manager.load_models()
        self.root.after(0, lambda: self.status_lbl.config(text="Models Ready", fg="#00ff00"))

    def toggle_cam(self):
        # 1. IDLE -> PREVIEW
        if self.state == "IDLE":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                self.state = "IDLE"
                return

            self.state = "PREVIEW"
            self.btn_action.config(text="SCAN SUBJECT", bg="#0055ff")
            # Start the video loop thread ONLY ONCE
            threading.Thread(target=self.loop).start()

        # 2. PREVIEW -> COUNTDOWN
        elif self.state == "PREVIEW":
            self.state = "COUNTDOWN"
            self.countdown_val = 3
            self.stabilizer.reset()
            self.btn_action.config(text="SCANNING...", state=tk.DISABLED, bg="#cc0000")
            self.update_labels(clear=True)
            self.root.after(1000, self.countdown_tick)

        # 3. DONE -> PREVIEW (Reset)
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
            if not ret: break
            frame = cv2.flip(frame, 1)

            display = frame.copy()
            h, w = frame.shape[:2]

            # --- STATE DRAWING LOGIC ---
            if self.state == "PREVIEW":
                cv2.putText(display, "READY - CLICK SCAN", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.draw_preview_face(display, frame)

            elif self.state == "COUNTDOWN":
                text = str(self.countdown_val)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
                cx, cy = w // 2 - tw // 2, h // 2 + th // 2
                cv2.putText(display, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

            elif self.state == "SCANNING":
                cv2.putText(display, "SCANNING... STAY STILL", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.analyze_frame(frame)
                if time.time() - self.scan_start_time > 2.5:
                    self.finish_scan()

            elif self.state == "DONE":
                cv2.putText(display, "COMPLETE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            cw = self.cam_lbl.winfo_width()
            ch = self.cam_lbl.winfo_height()
            if cw > 10: pil.thumbnail((cw, ch))
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
        # 1. Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.manager.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]

            H, W = frame.shape[:2]
            p = 20
            y1, x1 = max(0, y - p), max(0, x - p)
            y2, x2 = min(H, y + h + p), min(W, x + w + p)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                a, r = self.manager.predict_age_race(crop)
                g = self.manager.predict_gender(crop)
                t = self.manager.predict_tone(crop)
                h_style = self.manager.predict_hair(crop)

                self.stabilizer.update("age", a)
                self.stabilizer.update("race", r)
                self.stabilizer.update("gender", g)
                self.stabilizer.update("tone", t)
                self.stabilizer.update("hair", h_style)

        # 2. Clothing
        self.last_clothes = self.manager.predict_cloth(frame)

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
            self.cloth_list.delete(0, tk.END)
        else:
            # AGE LOGIC: CONVERT SINGLE NUMBER TO RANGE
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

            self.cloth_list.delete(0, tk.END)
            if hasattr(self, 'last_clothes'):
                for item in self.last_clothes:
                    self.cloth_list.insert(tk.END, f"• {item[0]}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()