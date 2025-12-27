# 🛡️ AI Room Guard — Intelligent Room Monitoring System with UI

**AI Room Guard** is an intelligent, voice-activated room security system that uses **facial recognition**, **speech interaction**, and **LLM-based decision-making** to monitor a room autonomously.  
It integrates **computer vision**, **speech recognition**, **text-to-speech**, and **Google Gemini AI** for real-time, human-like surveillance responses through a **Tkinter-based UI**.

## 📸 Key Features

- 🎙️ **Voice Activation** — Say *"Guard my room"* to start monitoring.
- 👁️ **Face Recognition** — Detects known and unknown individuals using DeepFace.
- 🤖 **AI Intruder Protocol** — Google Gemini decides whether a detected person’s speech is valid or suspicious using a 3-level escalation.
- 🔊 **Text-to-Speech Alerts** — Uses gTTS to respond to users or intruders.
- 🚨 **Intruder Alarm** — Plays a siren if escalation reaches level 3.
- 🪟 **Interactive UI** — Real-time logs, camera feed, and system state visualization.
 
## 🧩 System Architecture

```text
                ┌───────────────────────────────────────────────┐
                │                   User Interface               │
                │         (Tkinter + Live Camera Feed)           │
                │   ┌────────────────────────────────────────┐   │
                │   │ Video Frame │ Status │ Logs │ LLM Log │   │
                └───┴────────────────────────────────────────┘───┘
                                 │
                                 ▼
                ┌───────────────────────────────────────────────┐
                │              Core Logic Layer                 │
                │ - Guard Mode Activation (SpeechRecognition)   │
                │ - Face Recognition (DeepFace)                 │
                │ - Intruder Protocol (LLM + Speech I/O)        │
                │ - Text-to-Speech (gTTS + playsound)           │
                └───────────────────────────────────────────────┘
                                 │
                                 ▼
                ┌───────────────────────────────────────────────┐
                │            External Integrations              │
                │  Google Gemini API (LLM reasoning)            │
                │  DeepFace (face verification)                 │
                │  Microphone + Camera Hardware                 │
                └───────────────────────────────────────────────┘
````

## ⚙️ Installation & Setup

### 🧱 Prerequisites

Make sure you have **Python 3.9+** installed.

### 🧩 Required Libraries

Install all dependencies using:

```bash
pip install google-genai deepface SpeechRecognition pyaudio gTTS playsound opencv-python tf-keras tensorflow pillow
```

### 🔧 System Dependencies

Depending on your OS, install the following:

* **Windows / Mac**

  * `pip install pyaudio`
* **Linux (Debian / Ubuntu)**

  ```bash
  sudo apt install portaudio19-dev
  pip install pyaudio
  ```
* If `playsound` causes issues:

  ```bash
  pip install playsound==1.2.2
  ```

### 🔑 API Setup

1. Obtain a **Google Gemini API key** from [Google AI Studio](https://aistudio.google.com/).
2. Replace the placeholder in the code:

   ```python
   GEMINI_API_KEY = "YOUR_API_KEY_HERE"
   ```

### 🧍 Add Known Faces

Create a folder named `known_faces/` in the root directory, and place clear face images of authorized users there (e.g., `Utkarsh.jpg`, `friend.png`).


## 🚀 Running the Program

Run the script directly:

```bash
python gaurd_agent.py
```

Then:

1. Wait for the system to initialize.
2. Say **“guard my room”** to activate monitoring.
3. The camera feed and logs will appear in the UI window.


## 🧠 Integration Challenges & Solutions

| Challenge                        | Description                                                     | Solution                                                                                   |
| -------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Concurrent UI and Processing** | Tkinter UI would freeze during face recognition or audio tasks. | Implemented **threaded recognition and TTS operations** to ensure UI responsiveness.       |
| **Audio Device Conflicts**       | SpeechRecognition and playsound competed for the audio device.  | Used isolated threads and temporary file management for `gTTS` playback.                   |
| **DeepFace Performance Lag**     | High inference time during live recognition.                    | Used `opencv` backend with relaxed detection enforcement for lightweight real-time checks. |
| **Gemini API Rate Limits**       | Repeated intruder prompts could exceed API usage.               | Added cooldowns and conditional LLM invocations.                                           |
| **Cross-platform Audio Issues**  | `playsound` versions vary in stability.                         | Pinned stable version (`1.2.2`) and documented system-level dependencies.                  |

## ⚖️ Ethical Considerations

AI Room Guard is designed with **privacy, fairness, and user consent** in mind.

1. **Privacy Awareness**

   * No cloud storage; all facial data and recordings remain local.
   * Users must **consent to being recorded** in monitored spaces.

2. **Bias Mitigation**

   * DeepFace models can show demographic bias; users should retrain or calibrate using diverse datasets if deployed in real environments.

3. **Transparency**

   * Intruders are verbally informed when being monitored.
   * The system clearly announces recording and escalation actions.

4. **Responsible AI Use**

   * The Gemini API is used for reasoning only — no personal data is transmitted or stored remotely.


## 🧪 Testing Results

| Test Scenario                     | Expected Outcome                         | Result   |
| --------------------------------- | ---------------------------------------- | -------- |
| Known person detected             | User welcomed with name                  | ✅ Passed |
| Unknown person silent             | System escalates to level 3 with siren   | ✅ Passed |
| Unknown person gives valid reason | Gemini responds “VALID” → access granted | ✅ Passed |
| No microphone available           | Displays mic error in UI log             | ✅ Passed |
| No webcam                         | Graceful fallback with warning message   | ✅ Passed |


## 🖥️ UI Overview

| Section                 | Description                               |
| ----------------------- | ----------------------------------------- |
| **📹 Camera Feed**      | Live video stream from webcam             |
| **🛡️ Status Panel**    | Shows active/idle/alert system state      |
| **👥 Known Faces**      | Lists all enrolled users                  |
| **🎯 Detection Info**   | Displays recognition results in real time |
| **🤖 LLM Decision Log** | Shows Gemini model reasoning and verdicts |
| **📋 Activity Log**     | Timestamped system actions and alerts     |


## 🧾 License

This project is released under the **MIT License**.
You are free to modify and use it responsibly, with proper attribution.


## 💡 Future Enhancements

* Integration with IoT door locks or smart home devices.
* Mobile app companion for remote alerts.
* On-device LLM or offline fallback mode for edge security.
* Real-time database for access logs.

## 👨‍💻 Author

**Utkarsh Maurya | Chinmay Tripurwar**
Dual Degree Student, IIT Bombay
Project: *AI Room Guard – Intelligent Room Monitoring System*

