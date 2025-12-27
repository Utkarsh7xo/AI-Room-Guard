"""
AI Room Guard - Intelligent Room Monitoring System with UI

Required installations:
pip install google-genai deepface SpeechRecognition pyaudio gTTS playsound opencv-python tf-keras tensorflow pillow

Note: You may also need to install additional system dependencies:
- For audio: portaudio19-dev (Linux), PyAudio (Windows/Mac)
- For playsound: pip install playsound==1.2.2 (if latest version has issues)
"""

import os
import time
import cv2
import speech_recognition as sr
from gtts import gTTS
import playsound
from google import genai
from deepface import DeepFace
import tempfile
from pathlib import Path
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================
# Configuration
# ========================
GEMINI_API_KEY ="YOUR_API_KEY" # Replace with your actual API key
KNOWN_FACES_DIR = "known_faces"
ACTIVATION_PHRASE = "guard my room"
FACE_CHECK_INTERVAL = 3  # seconds between face recognition checks
WELCOME_COOLDOWN = 30  # seconds to wait before welcoming again
WELCOME_PAUSE = 30  # seconds to pause all monitoring after welcoming
RESPONSE_TIMEOUT = 10  # seconds to wait for intruder response
SIREN_SOUND_FILE = "siren.mp3"  # Path to siren sound file

# ========================
# Global Variables
# ========================
guard_mode_active = False
known_faces_db = []
last_welcome_time = {}
last_face_check_time = 0
ui_root = None
ui_elements = {}

# ========================
# UI Class
# ========================
class RoomGuardUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Room Guard")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Main container
        main_frame = tk.Frame(root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video label
        video_title = tk.Label(left_panel, text="📹 Camera Feed", font=("Arial", 14, "bold"), 
                              bg='#2d2d2d', fg='#00ff00')
        video_title.pack(pady=5)
        
        self.video_label = tk.Label(left_panel, bg='#000000')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right panel - Information
        right_panel = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Status section
        status_frame = tk.Frame(right_panel, bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(status_frame, text="🛡️ System Status", font=("Arial", 14, "bold"),
                bg='#2d2d2d', fg='#00ff00').pack()
        
        self.status_label = tk.Label(status_frame, text="⚪ IDLE", font=("Arial", 12, "bold"),
                                     bg='#2d2d2d', fg='#ffff00')
        self.status_label.pack(pady=5)
        
        # Known faces section
        faces_frame = tk.Frame(right_panel, bg='#2d2d2d')
        faces_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(faces_frame, text="👥 Known Faces", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#00bfff').pack()
        
        self.faces_label = tk.Label(faces_frame, text="None", font=("Arial", 10),
                                    bg='#2d2d2d', fg='#ffffff', wraplength=300)
        self.faces_label.pack(pady=5)
        
        # Detection info
        detection_frame = tk.Frame(right_panel, bg='#2d2d2d')
        detection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(detection_frame, text="🎯 Current Detection", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#ff69b4').pack()
        
        self.detection_label = tk.Label(detection_frame, text="No face detected", 
                                       font=("Arial", 10), bg='#2d2d2d', fg='#ffffff')
        self.detection_label.pack(pady=5)
        
        # LLM Response section
        llm_frame = tk.Frame(right_panel, bg='#2d2d2d')
        llm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(llm_frame, text="🤖 LLM Decision Log", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#ff8c00').pack()
        
        self.llm_text = scrolledtext.ScrolledText(llm_frame, height=8, font=("Courier", 9),
                                                  bg='#1a1a1a', fg='#00ff00', 
                                                  insertbackground='#00ff00')
        self.llm_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.llm_text.config(state=tk.DISABLED)
        
        # Activity log
        log_frame = tk.Frame(right_panel, bg='#2d2d2d')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(log_frame, text="📋 Activity Log", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#9370db').pack()
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Courier", 9),
                                                  bg='#1a1a1a', fg='#ffffff',
                                                  insertbackground='#ffffff')
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Control buttons
        button_frame = tk.Frame(right_panel, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.quit_button = tk.Button(button_frame, text="🚪 Quit", font=("Arial", 11, "bold"),
                                     bg='#ff4444', fg='#ffffff', command=self.quit_app,
                                     relief=tk.RAISED, bd=3)
        self.quit_button.pack(pady=5, fill=tk.X)
        
        global ui_elements
        ui_elements = {
            'video_label': self.video_label,
            'status_label': self.status_label,
            'faces_label': self.faces_label,
            'detection_label': self.detection_label,
            'llm_text': self.llm_text,
            'log_text': self.log_text
        }
    
    def quit_app(self):
        global guard_mode_active
        guard_mode_active = False
        self.root.quit()
        self.root.destroy()

import threading
import time
import cv2
# Assume other necessary imports like tkinter, os, deepface, etc., are present

def play_siren():
    """Plays the siren sound in a separate thread if the file exists."""
    def run():
        try:
            if os.path.exists(SIREN_SOUND_FILE):
                logger.critical("Playing siren!")
                update_ui_log("🚨 SIREN ACTIVATED 🚨", "CRITICAL")
                playsound.playsound(SIREN_SOUND_FILE)
            else:
                logger.error(f"Siren file not found: {SIREN_SOUND_FILE}")
                update_ui_log(f"✗ Siren file not found!", "ERROR")
        except Exception as e:
            logger.error(f"Could not play siren: {e}")
            update_ui_log(f"✗ Siren error: {e}", "ERROR")

    siren_thread = threading.Thread(target=run, daemon=True)
    siren_thread.start()
# ========================
# Global State Management
# ========================
# New state variable to prevent multiple recognition threads from running simultaneously
recognition_in_progress = False 

# ========================
# Face Recognition Processing (New Function)
# ========================
def process_face_recognition(frame, gemini_client):
    """
    Handles face recognition and subsequent actions in a separate thread 
    to avoid blocking the UI video feed.
    """
    global face_detected_last_check, last_welcome_time, recognition_in_progress
    
    try:
        name, face_detected = recognize_face(frame)
        current_time = time.time()
        
        if face_detected and name:
            if name not in last_welcome_time or \
               (current_time - last_welcome_time[name] > WELCOME_COOLDOWN):
                
                update_ui_detection(f"✓ Known: {name}", "#00ff00")
                speak(f"Welcome back, {name}. It's good to see you.")
                last_welcome_time[name] = current_time
                logger.info(f"Welcomed known user: {name}")
                update_ui_log(f"✓ Welcomed: {name}", "SUCCESS")
                
                logger.info(f"Pausing monitoring for {WELCOME_PAUSE} seconds")
                time.sleep(WELCOME_PAUSE) # This sleep now happens in the background
                face_detected_last_check = False
                      
        elif face_detected and not name:
            if not face_detected_last_check:
                logger.warning("Unknown face detected - starting intruder protocol")
                
                cv2.imwrite('last_intruder.jpg', frame)
                
                handle_intruder(gemini_client)
                face_detected_last_check = True
                
                logger.info("Pausing for 10 seconds after intruder interaction")
                time.sleep(10) # This sleep also happens in the background
            else:
                logger.debug("Same unknown face still detected")
        else:
            face_detected_last_check = False
            update_ui_detection("No face detected", "#ffffff")
            
    except Exception as e:
        logger.error(f"Error in recognition thread: {e}")
    finally:
        # IMPORTANT: Reset the flag so new recognitions can start
        recognition_in_progress = False

# ========================
# UI Update Functions
# ========================
def update_ui_log(message, level="INFO"):
    """Add message to activity log"""
    if 'log_text' in ui_elements:
        log_widget = ui_elements['log_text']
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_widget.config(state=tk.NORMAL)
        
        # Color coding
        if level == "WARNING":
            tag = "warning"
            log_widget.tag_config(tag, foreground="#ffaa00")
        elif level == "ERROR":
            tag = "error"
            log_widget.tag_config(tag, foreground="#ff4444")
        elif level == "CRITICAL":
            tag = "critical"
            log_widget.tag_config(tag, foreground="#ff0000", font=("Courier", 9, "bold"))
        elif level == "SUCCESS":
            tag = "success"
            log_widget.tag_config(tag, foreground="#00ff00")
        else:
            tag = "info"
            log_widget.tag_config(tag, foreground="#ffffff")
        
        log_widget.insert(tk.END, f"[{timestamp}] ", "info")
        log_widget.insert(tk.END, f"{message}\n", tag)
        log_widget.see(tk.END)
        log_widget.config(state=tk.DISABLED)

def update_ui_llm(user_response, llm_decision, level):
    """Update LLM response display"""
    if 'llm_text' in ui_elements:
        llm_widget = ui_elements['llm_text']
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        llm_widget.config(state=tk.NORMAL)
        llm_widget.insert(tk.END, f"\n{'='*50}\n", "separator")
        llm_widget.tag_config("separator", foreground="#666666")
        
        llm_widget.insert(tk.END, f"[{timestamp}] Level {level}\n", "time")
        llm_widget.tag_config("time", foreground="#00bfff")
        
        llm_widget.insert(tk.END, f"User: ", "label")
        llm_widget.tag_config("label", foreground="#ffff00", font=("Courier", 9, "bold"))
        llm_widget.insert(tk.END, f"{user_response}\n", "user")
        llm_widget.tag_config("user", foreground="#ffffff")
        
        llm_widget.insert(tk.END, f"LLM Decision: ", "label")
        
        if "VALID" in llm_decision:
            llm_widget.insert(tk.END, f"{llm_decision}\n", "valid")
            llm_widget.tag_config("valid", foreground="#00ff00", font=("Courier", 9, "bold"))
        else:
            llm_widget.insert(tk.END, f"{llm_decision}\n", "escalate")
            llm_widget.tag_config("escalate", foreground="#ff4444", font=("Courier", 9, "bold"))
        
        llm_widget.see(tk.END)
        llm_widget.config(state=tk.DISABLED)

def update_ui_status(status, color):
    """Update system status"""
    if 'status_label' in ui_elements:
        ui_elements['status_label'].config(text=status, fg=color)

def update_ui_detection(text, color="#ffffff"):
    """Update detection label"""
    if 'detection_label' in ui_elements:
        ui_elements['detection_label'].config(text=text, fg=color)

def update_ui_known_faces(faces):
    """Update known faces display"""
    if 'faces_label' in ui_elements:
        if faces:
            face_names = ", ".join([f['name'] for f in faces])
            ui_elements['faces_label'].config(text=f"{len(faces)} enrolled:\n{face_names}")
        else:
            ui_elements['faces_label'].config(text="None enrolled")

def update_video_frame(frame):
    """Update video feed in UI"""
    if 'video_label' in ui_elements and frame is not None:
        # Resize frame to fit UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        ui_elements['video_label'].imgtk = imgtk
        ui_elements['video_label'].configure(image=imgtk)

# ========================
# Initialize APIs
# ========================
def initialize_gemini():
    """Initialize Google Gemini API with new library"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini API initialized successfully")
        update_ui_log("✓ Gemini API initialized", "SUCCESS")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")
        update_ui_log(f"✗ Gemini API failed: {e}", "ERROR")
        return None

# ========================
# Text-to-Speech Function
# ========================
def speak(text):
    """Convert text to speech and play it"""
    try:
        logger.info(f"Speaking: {text}")
        update_ui_log(f"🔊 Speaking: {text}")
        tts = gTTS(text=text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            tts.save(temp_filename)
        
        playsound.playsound(temp_filename)
        
        try:
            os.unlink(temp_filename)
        except:
            pass
        
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        update_ui_log(f"TTS Error: {e}", "ERROR")

# ========================
# Speech Recognition Function
# ========================
def listen_for_speech(timeout=5, phrase_time_limit=10):
    """Listen for speech and return transcribed text"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logger.info("Listening...")
            update_ui_log("🎤 Listening for speech...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
        try:
            text = recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            update_ui_log(f"Heard: '{text}'", "INFO")
            return text.lower()
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            update_ui_log("Could not understand audio", "WARNING")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            update_ui_log(f"Speech service error: {e}", "ERROR")
            return None
            
    except sr.WaitTimeoutError:
        logger.info("Listening timeout - no speech detected")
        return None
    except Exception as e:
        logger.error(f"Microphone error: {e}")
        update_ui_log(f"Microphone error: {e}", "ERROR")
        return None

# ========================
# Face Enrollment
# ========================
def enroll_faces():
    """Enroll all known faces from the known_faces directory"""
    global known_faces_db
    
    if not os.path.exists(KNOWN_FACES_DIR):
        logger.warning(f"Creating {KNOWN_FACES_DIR} directory...")
        os.makedirs(KNOWN_FACES_DIR)
        update_ui_log(f"Created {KNOWN_FACES_DIR} folder", "WARNING")
        return []
    
    face_files = list(Path(KNOWN_FACES_DIR).glob('*.*'))
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    face_files = [f for f in face_files if f.suffix.lower() in valid_extensions]
    
    if not face_files:
        logger.warning(f"No image files found in {KNOWN_FACES_DIR}")
        update_ui_log(f"No images in {KNOWN_FACES_DIR}", "WARNING")
        return []
    
    enrolled = []
    logger.info(f"Enrolling {len(face_files)} known faces...")
    update_ui_log(f"Enrolling {len(face_files)} faces...")
    
    for face_file in face_files:
        try:
            name = face_file.stem
            logger.info(f"Enrolling: {name}")
            
            face_objs = DeepFace.extract_faces(img_path=str(face_file), 
                                               detector_backend='opencv',
                                               enforce_detection=False)
            
            if face_objs:
                enrolled.append({
                    'name': name,
                    'path': str(face_file)
                })
                logger.info(f"Successfully enrolled: {name}")
                update_ui_log(f"✓ Enrolled: {name}", "SUCCESS")
            else:
                logger.warning(f"No face detected in {face_file}")
                update_ui_log(f"✗ No face in {face_file}", "WARNING")
                
        except Exception as e:
            logger.error(f"Error enrolling {face_file}: {e}")
            update_ui_log(f"Error enrolling {face_file}: {e}", "ERROR")
    
    logger.info(f"Enrollment complete: {len(enrolled)} known faces")
    update_ui_log(f"Enrollment complete: {len(enrolled)} faces", "SUCCESS")
    return enrolled

# ========================
# Activation Listener
# ========================
def listen_for_activation():
    """Listen for the activation phrase"""
    global guard_mode_active
    
    logger.info(f"Waiting for activation phrase: '{ACTIVATION_PHRASE}'")
    update_ui_log(f"Say '{ACTIVATION_PHRASE}' to activate")
    speak(f"Say {ACTIVATION_PHRASE} to activate monitoring")
    
    while not guard_mode_active:
        try:
            text = listen_for_speech(timeout=5, phrase_time_limit=5)
            
            if text and ACTIVATION_PHRASE in text:
                guard_mode_active = True
                speak("Guard mode activated. Now monitoring the room.")
                logger.info("GUARD MODE ACTIVATED")
                update_ui_log("🛡️ GUARD MODE ACTIVATED", "SUCCESS")
                update_ui_status("🟢 ACTIVE - MONITORING", "#00ff00")
                return True
                
        except KeyboardInterrupt:
            logger.info("Activation listening interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error in activation listener: {e}")
            time.sleep(1)
    
    return True

# ========================
# Face Recognition
# ========================
def recognize_face(frame):
    """Recognize faces in the current frame"""
    global known_faces_db
    
    if not known_faces_db:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as fp:
                temp_frame = fp.name
                cv2.imwrite(temp_frame, frame)
            
            faces = DeepFace.extract_faces(img_path=temp_frame, 
                                           detector_backend='opencv',
                                           enforce_detection=False)
            
            os.unlink(temp_frame)
            
            if faces and len(faces) > 0:
                return None, True
            else:
                return None, False
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return None, False
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as fp:
            temp_frame = fp.name
            cv2.imwrite(temp_frame, frame)
        
        for known_face in known_faces_db:
            try:
                result = DeepFace.verify(
                    img1_path=temp_frame,
                    img2_path=known_face['path'],
                    model_name='VGG-Face',
                    detector_backend='opencv',
                    enforce_detection=False
                )
                
                if result['verified']:
                    os.unlink(temp_frame)
                    return known_face['name'], True
                    
            except Exception as e:
                logger.debug(f"Verification error with {known_face['name']}: {e}")
                continue
        
        faces = DeepFace.extract_faces(img_path=temp_frame, 
                                       detector_backend='opencv',
                                       enforce_detection=False)
        
        os.unlink(temp_frame)
        
        if faces and len(faces) > 0:
            return None, True
        else:
            return None, False
            
    except Exception as e:
        logger.error(f"Face recognition error: {e}")
        return None, False

# ========================
# Intruder Interaction
# ========================
def handle_intruder(gemini_client):
    """Handle interaction with unrecognized person using 3-level escalation"""
    
    system_prompt = """You are a helpful but firm AI security guard for a private room. 
Your goal is to identify and deter unauthorized individuals. You must follow a strict 3-level escalation protocol.
Analyze the person's response and determine if it's valid or if escalation is needed.

VALID responses include: stating they are a friend/visitor, giving a name, explaining they have permission, waiting for someone, etc.
ESCALATE for: refusing to answer, being hostile, giving nonsensical responses, or suspicious behavior.

Return ONLY one word: 'VALID' or 'ESCALATE'."""

    escalation_levels = [
        "Hello. I don't recognize you. Could you please state your name and purpose here?",
        "I cannot verify that information. This is a private area. Please leave immediately.",
        "You have not complied. I am now recording and the resident has been notified. This is your final warning to leave the premises."
    ]
    
    logger.warning("UNRECOGNIZED PERSON DETECTED - Starting intruder protocol")
    update_ui_log("⚠️ UNRECOGNIZED PERSON DETECTED", "CRITICAL")
    update_ui_status("🔴 ALERT - INTRUDER DETECTED", "#ff0000")
    update_ui_detection("⚠️ UNKNOWN PERSON", "#ff0000")
    
    for level, prompt in enumerate(escalation_levels, 1):
        logger.warning(f"Escalation Level {level}")
        update_ui_log(f"Escalation Level {level}/3", "WARNING")
        speak(prompt)
        
        response = listen_for_speech(timeout=RESPONSE_TIMEOUT, phrase_time_limit=15)
        
        if level == 3:
            play_siren()
            logger.critical("Final warning issued")
            update_ui_log("Final warning issued - Protocol complete", "CRITICAL")
            time.sleep(2)
            break
        
        if response:
            try:
                full_prompt = f"{system_prompt}\n\nIntruder's response: '{response}'\n\nYour decision:"
                
                llm_response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                
                decision = llm_response.text.strip().upper()
                
                logger.info(f"LLM Decision: {decision}")
                update_ui_llm(response, decision, level)
                
                if 'VALID' in decision:
                    speak("Thank you for your cooperation. Please proceed, but note that this interaction has been logged.")
                    logger.info("Intruder provided valid response")
                    update_ui_log("✓ Valid response - Access granted", "SUCCESS")
                    update_ui_status("🟢 ACTIVE - MONITORING", "#00ff00")
                    return
                    
            except Exception as e:
                logger.error(f"LLM error: {e}")
                update_ui_log(f"LLM error: {e}", "ERROR")
        else:
            logger.warning("No response received from intruder")
            update_ui_log("No response from person", "WARNING")
        
        time.sleep(1)
    
    logger.critical("Intruder protocol completed - all warnings issued")
    update_ui_log("Protocol complete - All warnings issued", "CRITICAL")
    update_ui_status("🟡 ACTIVE - MONITORING", "#ffaa00")

# ========================
# Main Monitoring Loop
# ========================
def monitor_room(gemini_client):
    """Main room monitoring function"""
    global guard_mode_active, last_face_check_time, last_welcome_time,recognition_in_progress,face_detected_last_check
    
    cap = None
    for camera_index in range(3):
        try:
            logger.info(f"Trying to open camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Successfully opened camera at index {camera_index}")
                    update_ui_log(f"✓ Camera {camera_index} opened", "SUCCESS")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
                
        except Exception as e:
            logger.error(f"Error trying camera index {camera_index}: {e}")
            cap = None
    
    if cap is None or not cap.isOpened():
        logger.error("Cannot open any webcam")
        update_ui_log("✗ Cannot access webcam", "ERROR")
        speak("Error: Cannot access webcam. Please check your camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    logger.info("Starting room monitoring...")
    update_ui_log("Starting room monitoring...", "SUCCESS")
    speak("Camera is now active. Monitoring started.")
    
    face_detected_last_check = False
    
    try:
        while guard_mode_active:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # This now runs continuously without being blocked
            update_video_frame(frame)
            
            current_time = time.time()

            # Check if it's time for face recognition AND if a check isn't already running
            if not recognition_in_progress and (current_time - last_face_check_time >= FACE_CHECK_INTERVAL):
                last_face_check_time = current_time
                recognition_in_progress = True

                # Create and start the background thread for face recognition
                # Pass a copy of the frame to the thread
                recognition_thread = threading.Thread(
                    target=process_face_recognition, 
                    args=(frame.copy(), gemini_client), 
                    daemon=True
                )
                recognition_thread.start()
            
            # Small delay for UI responsiveness
            ui_root.update()
            time.sleep(0.03)
                
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        update_ui_log("Monitoring stopped by user", "WARNING")
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
        update_ui_log(f"Error: {e}", "ERROR")
    finally:
        if cap is not None:
            cap.release()
        logger.info("Monitoring stopped")
        update_ui_log("Monitoring stopped", "WARNING")
        update_ui_status("⚪ STOPPED", "#ffff00")

# ========================
# Main Function
# ========================
def main():
    """Main entry point"""
    global ui_root
    
    logger.info("=== AI Room Guard Starting ===")
    
    # Create UI
    ui_root = tk.Tk()
    app = RoomGuardUI(ui_root)
    
    update_ui_log("=== AI Room Guard Starting ===", "INFO")
    update_ui_status("⚪ INITIALIZING", "#ffff00")
    
    # Initialize Gemini in separate thread
    def init_and_run():
        gemini_client = initialize_gemini()
        if not gemini_client:
            logger.error("Cannot proceed without Gemini API")
            update_ui_log("✗ Gemini API required", "ERROR")
            speak("Error: Gemini API initialization failed")
            return
        
        global known_faces_db
        known_faces_db = enroll_faces()
        update_ui_known_faces(known_faces_db)
        
        if not known_faces_db:
            logger.warning("No known faces enrolled")
            update_ui_log("No known faces - intruder detection only", "WARNING")
            speak("No known faces found. Only intruder detection will be active.")
        else:
            speak(f"{len(known_faces_db)} known faces enrolled successfully.")
        
        speak("AI Room Guard initialized.")
        update_ui_status("⚪ IDLE - WAITING", "#ffff00")
        
        if listen_for_activation():
            monitor_room(gemini_client)
        
        speak("Guard mode deactivated. Shutting down.")
        update_ui_log("=== AI Room Guard Shutdown ===", "INFO")
        logger.info("=== AI Room Guard Shutdown ===")
    
    # Run initialization
    # Run initialization in background thread
    init_thread = threading.Thread(target=init_and_run, daemon=True)
    init_thread.start()
    
    # Start UI main loop
    ui_root.mainloop()

if __name__ == "__main__":

    main()
