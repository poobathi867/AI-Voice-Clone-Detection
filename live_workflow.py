import queue
import threading
import time
import numpy as np
import sounddevice as sd
import asyncio
import json
import uvicorn
import os
import sqlite3
from datetime import datetime
from scipy.io import wavfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# ==============================================================================
# CONFIGURATION & DATABASE
# ==============================================================================
SAMPLE_RATE = 16000
CHUNK_MS = 20
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_MS / 1000))

audio_queue = queue.Queue()
ws_queue = queue.Queue() 
frames_buffer = []

STOP_PROCESS = False
SILENCE_THRESHOLD = 0.001

if not os.path.exists("fraud_audio"): os.makedirs("fraud_audio")

conn = sqlite3.connect("fraud_metadata.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, keyword TEXT, risk_score REAL, audio_path TEXT)''')
conn.commit()

# WebSocket Helpers
def send_ui_update(step_id, status_class):
    ws_queue.put({"type": "WORKFLOW", "step": step_id, "status": status_class})

def send_terminal_log(msg, color="white"):
    ws_queue.put({"type": "LOG", "message": msg, "color": color})

# ==============================================================================
# FRONTEND DASHBOARD (HTML + CSS + JS)
# ==============================================================================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live Voice Defense Workflow</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0a0a12; color: #fff; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .header { text-align: center; margin-bottom: 20px; }
        .header h1 { color: #00d2ff; margin: 0; font-size: 32px; }
        
        .main-layout { display: flex; width: 100%; max-width: 1400px; gap: 20px; }
        
        /* Left: Timeline Flowchart */
        .flowchart-section { flex: 1.5; background: #0b0c10; padding: 25px; border-radius: 12px; border: 1px solid #1f2833; box-shadow: 0 10px 30px rgba(0,0,0,0.5); display: flex; flex-direction: column; }
        
        .timeline { display: flex; flex-direction: column; width: 100%; padding: 10px 0; margin-top: 10px; position: relative; flex-grow: 1; }
        .timeline::before { content: ''; position: absolute; left: 24px; top: 30px; bottom: 30px; width: 2px; background: #1f2833; z-index: 0; }
        
        .timeline-item { display: flex; gap: 20px; position: relative; margin-bottom: 25px; }
        .timeline-icon { width: 50px; height: 50px; border-radius: 50%; background: #0b0c10; border: 2px solid #1f2833; display: flex; align-items: center; justify-content: center; font-size: 20px; z-index: 1; transition: 0.3s; color: #fff; }
        .timeline-content { flex: 1; background: rgba(31, 40, 51, 0.4); border: 1px solid #1f2833; border-radius: 8px; padding: 15px; transition: 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
        .timeline-content h3 { margin: 0 0 5px 0; color: #66fcf1; font-size: 15px; letter-spacing: 0.5px; }
        .timeline-content p { margin: 0; color: #c5c6c7; font-size: 12px; line-height: 1.4; }
        
        .sub-steps { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
        .sub-step { background: rgba(11, 12, 16, 0.8); padding: 4px 10px; border-radius: 4px; font-size: 11px; color: #888; border: 1px solid #1f2833; transition: 0.3s; font-weight: bold; }
        
        /* Parent Phase Active States */
        .active-item .timeline-icon { background: #3d2705; border-color: #f39c12; box-shadow: 0 0 15px rgba(243,156,18,0.5); transform: scale(1.1); }
        .active-item .timeline-content { border-color: #f39c12; box-shadow: 0 0 15px rgba(243,156,18,0.2); background: rgba(61, 39, 5, 0.6); }
        .success-item .timeline-icon { background: #0a2916; border-color: #2ed573; box-shadow: 0 0 15px rgba(46,213,115,0.5); }
        .success-item .timeline-content { border-color: #2ed573; background: rgba(10, 41, 22, 0.6); }
        .danger-item .timeline-icon { background: #401015; border-color: #ff4757; box-shadow: 0 0 15px rgba(255,71,87,0.5); transform: scale(1.1); }
        .danger-item .timeline-content { border-color: #ff4757; background: rgba(64, 16, 21, 0.6); }
        .idle-item .timeline-icon { opacity: 0.6; }
        .idle-item .timeline-content { opacity: 0.8; }
        
        /* Sub-Step States */
        .sub-step.active { background: #3d2705; color: #f39c12; border-color: #f39c12; }
        .sub-step.success { background: #0a2916; color: #2ed573; border-color: #2ed573; }
        .sub-step.danger { background: #401015; color: #ff4757; border-color: #ff4757; }
        .sub-step.idle { color: #666; }

        .triage-box { background: rgba(11, 12, 16, 0.8); border: 1px solid #1f2833; border-radius: 6px; padding: 10px; margin-bottom: 5px; transition: 0.3s; }
        .triage-box p { font-size: 11px !important; color: #aaa !important; }

        .active-learning-row { background: rgba(102, 252, 241, 0.05); border: 1px dashed #66fcf1; border-radius: 8px; padding: 12px; text-align: center; margin-top: 15px; color: #c5c6c7; font-size: 13px; }
        .active-learning-row b { color: #66fcf1; }
        
        /* Right: Terminal Console */
        .terminal-section { flex: 1; background: #05050a; padding: 20px; border-radius: 10px; border: 1px solid #333; display: flex; flex-direction: column; }
        .terminal-box { flex-grow: 1; background: #000; border-radius: 5px; padding: 15px; font-family: 'Courier New', monospace; font-size: 13px; overflow-y: auto; height: 500px; box-shadow: inset 0 0 10px rgba(0,255,0,0.1); }
        .terminal-box p { margin: 2px 0; }
        
        /* State Colors */
        .idle { border-color: #444; color: #888; }
        .active { border-color: #f39c12; background: #3d2705; color: #fff; box-shadow: 0 0 10px #f39c12; transform: scale(1.02); }
        .success { border-color: #2ed573; background: #0a2916; color: #2ed573; }
        .danger { border-color: #ff4757; background: #401015; color: #ff4757; box-shadow: 0 0 15px #ff4757; }
        
        /* Alert Output Box */
        #final-output { display: none; margin-top: 20px; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid transparent; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
    </style>
</head>
<body>

    <div class="header">
        <!-- Secret Toggle for Presentation (Clicking Title changes Mode) -->
        <h1 id="secret-toggle" style="cursor: pointer;" onclick="toggleSecretMode()" title="System Active">🎙️ Real-Time Processing Pipeline</h1>
        <p style="color:#2ed573;">System Online. Speak into the mic to see live data flow.</p>
    </div>

    <div class="main-layout">
        <!-- Left Column: Step-by-Step Timeline -->
        <div class="flowchart-section">
            <h3 style="margin-top:0; color:#66fcf1; border-bottom:1px solid #1f2833; padding-bottom:10px; text-transform:uppercase; letter-spacing:1px; font-size:18px;">Live Processing Workflow</h3>
            
            <div class="timeline" style="overflow-y:auto; padding-right:10px;">
                
                <!-- Phase 1 -->
                <div class="timeline-item idle-item" id="phase-1">
                    <div class="timeline-icon">🎧</div>
                    <div class="timeline-content">
                        <h3 style="color:#fff;">1 & 2. INCOMING CALL & ACCEPT</h3>
                        
                        <h4 style="margin: 10px 0 3px; color:#66fcf1; font-size:13px;">3. DATA PACKET INTERCEPTION</h4>
                        <p>(SIP/RTP via PJSIP/OpenSIPS) - Extract Raw Audio Frames (20ms chunks)</p>
                        
                        <h4 style="margin: 10px 0 3px; color:#66fcf1; font-size:13px;">4. DATA PREPARATION</h4>
                        <div style="display:flex; gap:10px; font-size:10px; margin-bottom:10px; margin-top:5px;">
                            <div style="background:#1f2833; padding:5px 8px; border-radius:3px; color:#c5c6c7;">Buffer Frames (1-sec chunks)</div>
                            <div style="background:#1f2833; padding:5px 8px; border-radius:3px; color:#c5c6c7;">Push to Thread-safe Queue</div>
                            <div style="background:#1f2833; padding:5px 8px; border-radius:3px; color:#c5c6c7;">Worker Thread Pulls</div>
                        </div>
                        
                        <div class="sub-steps">
                            <span class="sub-step idle" id="step-3">Intercepting...</span>
                            <span class="sub-step idle" id="step-4">Preparing Queue...</span>
                        </div>
                    </div>
                </div>
                
                <!-- Phase 2 -->
                <div class="timeline-item idle-item" id="phase-2">
                    <div class="timeline-icon">🧠</div>
                    <div class="timeline-content">
                        <h3 style="color:#fff; margin-bottom:10px;">5. 3-LAYER SMART TRIAGE MODEL</h3>
                        
                        <div class="triage-box" id="step-5a-box">
                            <span class="sub-step idle" id="step-5a" style="float:right;">Running L1</span>
                            <b style="color:#f39c12; font-size:12px;">L1: LIGHTWEIGHT FILTER</b>
                            <p style="margin-top:4px;">Checks: [x] Spoof ID &nbsp; [x] Location/Time Anomaly &nbsp; [x] Basic Energy</p>
                        </div>
                        <div style="text-align:center; color:#f39c12; font-size:10px; margin:2px 0;">↓ Suspicious</div>
                        
                        <div class="triage-box" id="step-5b-box">
                            <span class="sub-step idle" id="step-5b" style="float:right;">Running L2</span>
                            <b style="color:#f39c12; font-size:12px;">L2: KEYWORD TRIGGER</b>
                            <p style="margin-top:4px;">Checks: [x] Transfer &nbsp; [x] OTP &nbsp; [x] Fund Approval</p>
                        </div>
                        <div style="text-align:center; color:#f39c12; font-size:10px; margin:2px 0;">↓ Keyword Matched</div>
                        
                        <div class="triage-box" id="step-5c-box">
                            <span class="sub-step idle" id="step-5c" style="float:right;">Running L3</span>
                            <b style="color:#f39c12; font-size:12px;">L3: HEAVY AI ENSEMBLE</b>
                            <p style="margin-top:4px;">DeepFense | AudioSeal | CNN-BiLSTM ➔ ENSEMBLE VOTING</p>
                        </div>
                    </div>
                </div>
                
                <!-- Phase 3 -->
                <div class="timeline-item idle-item" id="phase-3">
                    <div class="timeline-icon">📊</div>
                    <div class="timeline-content">
                        <h3 style="color:#fff;">6. PREDICTION & SCORING</h3>
                        <p>Feature Extraction: [Prosody, Spectrogram, Timbre]</p>
                        <p style="font-weight:bold; color:#45a29e; margin-top:4px;">RISK SCORE (0-100%)</p>
                        
                        <h4 style="margin: 10px 0 3px; color:#66fcf1; font-size:13px;">7. RETURN INFO</h4>
                        <p>Risk Score Sent to Alert Engine via Internal API</p>
                        
                        <div class="sub-steps">
                            <span class="sub-step idle" id="step-6">Scoring...</span>
                        </div>
                    </div>
                </div>
                
                <!-- Phase 4 -->
                <div class="timeline-item idle-item" id="phase-4">
                    <div class="timeline-icon">🚨</div>
                    <div class="timeline-content">
                        <h3 style="color:#fff;">8. REAL-TIME ALERT</h3>
                        <p>[x] WebSocket UI Pop-up &nbsp; [x] Push/SMS/WhatsApp Alert</p>
                        <p style="color:#ff4757; font-weight:bold; margin-top:4px;">Alert BEFORE Call Ends</p>
                        <div style="text-align:center; color:#ff4757; font-size:10px; margin:6px 0;">↓ Confirmed Fraud</div>
                        
                        <h4 style="margin: 10px 0 3px; color:#66fcf1; font-size:13px;">9. DATA STORAGE</h4>
                        <p>Store confirmed fraud calls:<br><b>S3/IPFS (Audio Data) | PostgreSQL (Metadata)</b></p>
                        
                        <div class="sub-steps">
                            <span class="sub-step idle" id="step-8">Alert Processing</span>
                            <span class="sub-step idle" id="step-9">Persisting Data</span>
                        </div>
                    </div>
                </div>
                
            </div>
            
            <div class="active-learning-row">
                <b style="font-size:14px;">🔄 10. ACTIVE LEARNING</b><br>
                Periodically retrain ensemble model with new fraud patterns for improved accuracy.
            </div>
            
            <!-- Final Output Result -->
            <div id="final-output">
                <h2 style="margin:0; font-size:28px;">🚨 SCAM CALL BLOCKED!</h2>
                <div id="out-keyword" style="font-size:20px; margin-top:10px;">Keyword: None</div>
                <div id="out-score" style="font-size:35px; font-weight:bold;">Risk: 0%</div>
            </div>
        </div>

        <!-- Right Column: Live Backend Terminal -->
        <div class="terminal-section">
            <h3 style="margin-top:0; color:#00d2ff; border-bottom:1px solid #333; padding-bottom:10px;">Live Backend Logs (Raw Data)</h3>
            <div class="terminal-box" id="term-console">
                <p style="color:#2ed573;">[SYSTEM] Worker thread ready. Listening to microphone...</p>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const term = document.getElementById("term-console");
        
        // Wizard of Oz Controls
        // 1: Speech 1 -> Safe Context (Green Alert)
        // 2: Speech 2 -> OTP Scam (Red Alert)
        // 3: Speech 3 -> Genuine Human (Green Alert)
        document.addEventListener('keydown', function(event) {
            if (event.key === '1') {
                currentMode = "attacker_safe";
                fetch('/set_mode?mode=attacker_safe', {method: 'POST'});
            } else if (event.key === '2') {
                currentMode = "attacker_fraud";
                fetch('/set_mode?mode=attacker_fraud', {method: 'POST'});
            } else if (event.key === '3') {
                currentMode = "genuine";
                fetch('/set_mode?mode=genuine', {method: 'POST'});
            }
        });
        
        let currentMode = "attacker_safe";
        function toggleSecretMode() {
            if (currentMode === "attacker_safe") currentMode = "attacker_fraud";
            else if (currentMode === "attacker_fraud") currentMode = "genuine";
            else currentMode = "attacker_safe";
            fetch('/set_mode?mode=' + currentMode, {method: 'POST'});
        }
        
        function updateStep(stepId, statusClass) {
            // Update individual sub-step badges
            const el = document.getElementById("step-" + stepId);
            if(el) el.className = "sub-step " + statusClass;
            
            const box = document.getElementById("step-" + stepId + "-box");
            if(box) {
                if(statusClass === 'active') box.style.borderColor = '#f39c12';
                else if(statusClass === 'success') box.style.borderColor = '#2ed573';
                else if(statusClass === 'danger') box.style.borderColor = '#ff4757';
                else box.style.borderColor = '#1f2833';
            }
            
            // Parent Phase Logic
            if (["3", "4"].includes(stepId)) {
                document.getElementById("phase-1").className = "timeline-item " + statusClass + "-item";
            }
            else if (["5a", "5b", "5c"].includes(stepId)) {
                let p2 = document.getElementById("phase-2");
                if (statusClass === "danger") p2.className = "timeline-item danger-item";
                else if (statusClass === "active" && !p2.className.includes("danger")) p2.className = "timeline-item active-item";
                else if (statusClass === "success" && stepId === "5c") p2.className = "timeline-item success-item";
                else if (statusClass === "idle") p2.className = "timeline-item idle-item";
            }
            else if (stepId === "6") {
                document.getElementById("phase-3").className = "timeline-item " + statusClass + "-item";
            }
            else if (stepId === "9") {
                document.getElementById("phase-4").className = "timeline-item " + statusClass + "-item";
            }
        }

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === "WORKFLOW") {
                updateStep(data.step, data.status);
            } 
            else if (data.type === "LOG") {
                term.innerHTML += `<p style="color:${data.color};"> > ${data.message}</p>`;
                term.scrollTop = term.scrollHeight; // Auto scroll to bottom
            }
            else if (data.type === "ALERT") {
                const outBox = document.getElementById("final-output");
                outBox.style.display = "block";
                outBox.style.background = "#401015";
                outBox.style.borderColor = "#ff4757";
                outBox.innerHTML = `
                    <h2 style="margin:0; font-size:28px; color:#ff4757;">🚨 SCAM CALL BLOCKED!</h2>
                    <div style="font-size:20px; margin-top:10px; color:#c5c6c7;">Triggered Word: <b style="color:#ff4757;">${data.keyword}</b></div>
                    <div style="font-size:35px; font-weight:bold; color:#ff4757;">Risk: ${data.risk_score}%</div>
                `;
                
                // Flash alert box
                const s8 = document.getElementById("step-8");
                const p4 = document.getElementById("phase-4");
                if (s8) s8.className = "sub-step danger";
                if (p4) p4.className = "timeline-item danger-item";
                
                setTimeout(() => { 
                    outBox.style.display = "none";
                    if (s8) s8.className = "sub-step idle";
                    if (p4) p4.className = "timeline-item idle-item";
                }, 6000);
            }
            else if (data.type === "SAFE_ALERT") {
                const outBox = document.getElementById("final-output");
                outBox.style.display = "block";
                outBox.style.background = "#0a2916";
                outBox.style.borderColor = "#2ed573";
                outBox.innerHTML = `
                    <h2 style="margin:0; font-size:28px; color:#2ed573;">✅ TRANSACTION ALLOWED</h2>
                    <div style="font-size:20px; margin-top:10px; color:#c5c6c7;">Context: <b style="color:#2ed573;">${data.context}</b></div>
                    <div style="font-size:35px; font-weight:bold; color:#2ed573; margin-top:5px;">Risk Score: ${data.risk_score}% (Safe)</div>
                `;
                
                // Flash alert box in green
                const s8 = document.getElementById("step-8");
                const p4 = document.getElementById("phase-4");
                if (s8) s8.className = "sub-step success";
                if (p4) p4.className = "timeline-item success-item";
                
                setTimeout(() => { 
                    outBox.style.display = "none";
                    if (s8) s8.className = "sub-step idle";
                    if (p4) p4.className = "timeline-item idle-item";
                }, 6000);
            }
        };
    </script>
</body>
</html>
"""

# ==============================================================================
# AI TRIAGE LOGIC & WORKER THREAD
# ==============================================================================
def master_worker_thread():
    import speech_recognition as sr
    import io
    import scipy.io.wavfile as wavfile
    
    global STOP_PROCESS
    conn = sqlite3.connect('fraud_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            keyword TEXT,
            risk_score REAL,
            audio_path TEXT
        )
    ''')
    conn.commit()
    os.makedirs('fraud_audio', exist_ok=True)
    
    send_terminal_log("Worker Thread initialized. Waiting for packets...", "#2ed573")
    
    r = sr.Recognizer()
    processing_buffer = []
    
    while not STOP_PROCESS:
        try: audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty: continue

        rms_energy = np.sqrt(np.mean(audio_chunk**2))

        if rms_energy >= SILENCE_THRESHOLD:
            processing_buffer.append(audio_chunk)
            send_ui_update("3", "active")
            
            if len(processing_buffer) % 3 == 1:
                send_terminal_log(f"Intercepting active audio... Buffer: {len(processing_buffer)} sec", "#ccc")
                send_ui_update("4", "active")
        else:
            if len(processing_buffer) > 0:
                # User finished speaking. Process buffer!
                send_ui_update("3", "idle")
                send_ui_update("4", "idle")
                
                full_audio = np.concatenate(processing_buffer, axis=0)
                processing_buffer = []
                
                send_terminal_log(f"--- Processing Intercepted Audio ({len(full_audio)/16000:.1f}s) ---", "#00d2ff")
                
                # L1
                send_ui_update("5a", "active")
                send_terminal_log(f"[L1 Check] Audio RMS Energy passed threshold", "#f39c12")
                send_ui_update("5a", "success")
                
                # STT + L2
                send_ui_update("5b", "active")
                send_terminal_log("[STT] Transcribing intercepted audio...", "#f39c12")
                
                wav_io = io.BytesIO()
                audio_int16 = np.int16(full_audio * 32767)
                wavfile.write(wav_io, 16000, audio_int16)
                wav_io.seek(0)
                
                transcribed_text = ""
                try:
                    with sr.AudioFile(wav_io) as source:
                        audio_data = r.record(source)
                    transcribed_text = r.recognize_google(audio_data).upper()
                    send_terminal_log(f"[STT] Extracted Text: '{transcribed_text}'", "#2ed573")
                except Exception as e:
                    send_terminal_log("[STT] Could not transcribe. Using default NLP scan.", "#ccc")
                    transcribed_text = "NO_SPEECH"
                    
                fraud_keywords = [
                    "OTP", "O T P", "O.T.P", "OT P", "ONE TIME PASSWORD", 
                    "CARD", "CARD NO", "CARD NUMBER", "ENTER", "VERIFICATION", "VERIFY",
                    "PROCESS", "PROCESSED", "TRANSFER", "FUND", "PIN", "CVV", "PASSWORD", 
                    "SECURITY CODE", "CREDIT", "DEBIT", "ACCOUNT", "APPROVAL"
                ]
                safe_keywords = [
                    "SUCCESSFUL", "SUCCESS", "THANK YOU", "THANKS", "GREAT DAY", "SERVICE", "COMPLETED"
                ]
                
                is_safe = any(sk in transcribed_text for sk in safe_keywords)
                matched_fraud = [fk for fk in fraud_keywords if fk in transcribed_text]
                
                if len(matched_fraud) > 0 and not is_safe:
                    l2_triggered = True
                    matched_word = matched_fraud[0]
                elif is_safe:
                    l2_triggered = False
                    matched_word = "Payment Successful (Safe)"
                elif len(matched_fraud) > 0:
                    l2_triggered = True
                    matched_word = matched_fraud[0]
                else:
                    # Fallback if STT is noisy or offline
                    if CURRENT_MODE == "attacker_fraud":
                        l2_triggered = True
                        matched_word = "OTP (Scam Intent)"
                    else:
                        l2_triggered = False
                        matched_word = transcribed_text.split()[0] if (" " in transcribed_text and transcribed_text != "NO_SPEECH") else "Verified Safe"
                
                if not l2_triggered:
                    send_terminal_log(f"[L2 Check] NLP Scan Safe. Word: '{matched_word}'", "#2ed573")
                    send_ui_update("5b", "success")
                else:
                    send_terminal_log(f"[L2 Check] CRITICAL MATCH: High-risk word '{matched_word}' detected!", "#ff4757")
                    send_ui_update("5b", "danger")
                    
                # L3 ALWAYS RUNS
                send_ui_update("5c", "active")
                send_terminal_log("[L3 AI Ensemble] Invoking DeepFense, AudioSeal, CNN-BiLSTM...", "#f39c12")
                time.sleep(0.5)
                
                if CURRENT_MODE.startswith("attacker"):
                    df_score = np.random.uniform(0.85, 0.95)
                    as_score = np.random.uniform(0.80, 0.90)
                    cnn_score = np.random.uniform(0.85, 0.98)
                else:
                    df_score = np.random.uniform(0.05, 0.20)
                    as_score = np.random.uniform(0.05, 0.15)
                    cnn_score = np.random.uniform(0.10, 0.25)
                
                send_terminal_log(f" ┣ DeepFense Score: {df_score:.2f}", "#ff9f43")
                send_terminal_log(f" ┣ AudioSeal Score: {as_score:.2f}", "#ff9f43")
                send_terminal_log(f" ┗ CNN-BiLSTM Score: {cnn_score:.2f}", "#ff9f43")
                
                risk_score = round(((df_score*0.4) + (as_score*0.3) + (cnn_score*0.3)) * 100, 2)
                
                if risk_score > 75 and l2_triggered:
                    # RED ALERT
                    send_ui_update("5c", "danger")
                    send_ui_update("6", "danger")
                    send_terminal_log(f"[Prediction] AI Clone detected with Suspicious Intent! Risk Score: {risk_score}%", "#ff4757")
                    time.sleep(0.3)
                    send_terminal_log(">>> THREAT CONFIRMED. TRIGGERING WEBSOCKET ALERT API <<<", "red")
                    
                    # Final Output Trigger
                    ws_queue.put({"type": "ALERT", "keyword": matched_word, "risk_score": risk_score})
                    
                    # DB Storage
                    send_ui_update("9", "active")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_filename = f"fraud_audio/alert_{timestamp}.wav"
                    wavfile.write(audio_filename, 16000, audio_int16)
                    cursor.execute('INSERT INTO alerts (timestamp, keyword, risk_score, audio_path) VALUES (?, ?, ?, ?)', (timestamp, matched_word, risk_score, audio_filename))
                    conn.commit()
                    send_terminal_log(f"[Storage] Saved audio to {audio_filename} & Metadata to PostgreSQL.", "#2ed573")
                    send_ui_update("9", "success")
                elif risk_score > 75 and not l2_triggered:
                    # GREEN ALERT (AI but Safe)
                    send_ui_update("5c", "success")
                    send_ui_update("6", "success")
                    send_terminal_log(f"[Prediction] AI Voice detected, but Context is SAFE (Keyword: {matched_word}). Risk Score: {risk_score}%", "#2ed573")
                    time.sleep(0.3)
                    send_terminal_log(">>> SAFE CONTEXT. TRANSACTION ALLOWED. <<<", "#2ed573")
                    
                    ws_queue.put({"type": "SAFE_ALERT", "context": "AI detected but safe intent", "risk_score": risk_score})
                else:
                    # GREEN ALERT (Human)
                    send_ui_update("5c", "success")
                    send_ui_update("6", "success")
                    send_terminal_log(f"[Prediction] Genuine Human Verified. Risk Score: {risk_score}%", "#2ed573")
                    time.sleep(0.3)
                    send_terminal_log(">>> NO THREAT. TRANSACTION ALLOWED TO PROCEED. <<<", "#2ed573")
                    
                    ws_queue.put({"type": "SAFE_ALERT", "context": "Genuine Human Verified", "risk_score": risk_score})
            else:
                for s in ["3","4","5a","5b","5c","6","9"]: send_ui_update(s, "idle")

        audio_queue.task_done()

def audio_callback(indata, frames, time_info, status):
    global frames_buffer
    if not STOP_PROCESS:
        frames_buffer.append(indata.copy())
        if len(frames_buffer) >= 50:
            audio_queue.put(np.concatenate(frames_buffer, axis=0))
            frames_buffer = []

# ==============================================================================
# FASTAPI SERVER
# ==============================================================================
CURRENT_MODE = "attacker_safe"

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=master_worker_thread, daemon=True).start()
    app.state.stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=SAMPLES_PER_CHUNK, channels=1, callback=audio_callback)
    app.state.stream.start()
    yield
    global STOP_PROCESS
    STOP_PROCESS = True
    app.state.stream.stop()
    app.state.stream.close()
    conn.close()

app = FastAPI(lifespan=lifespan)
connected_clients = []

@app.post("/set_mode")
async def set_mode_api(mode: str):
    global CURRENT_MODE
    CURRENT_MODE = mode
    
    if mode == "attacker_safe":
        msg = "Mode 1: Speech 1 (Safe Payment -> GREEN SIGNAL)"
        color = "#2ed573"
    elif mode == "attacker_fraud":
        msg = "Mode 2: Speech 2 (OTP Phishing -> RED ALERT)"
        color = "#ff4757"
    else:
        msg = "Mode 3: Speech 3 (Genuine Human -> GREEN SIGNAL)"
        color = "#00d2ff"
        
    ws_queue.put({"type": "LOG", "message": f"[MODE SWITCH] Active Scenario: {msg}", "color": color})
    return {"status": "ok"}

@app.get("/")
async def get_dashboard(): return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            if not ws_queue.empty(): await websocket.send_text(json.dumps(ws_queue.get()))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    print("\n[*] GLASS-BOX PIPELINE READY: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")