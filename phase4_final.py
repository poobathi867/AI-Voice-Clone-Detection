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

# -------------------------------------------------------------
# 1. CONFIGURATION & DATABASE SETUP
# -------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_MS = 20
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_MS / 1000))

audio_queue = queue.Queue()
frames_buffer = []
alert_queue = queue.Queue()

STOP_PROCESS = False
SILENCE_THRESHOLD = 0.001
SILENT_CHUNKS_LIMIT = 2

# Create folder for S3/IPFS simulation
if not os.path.exists("fraud_audio"):
    os.makedirs("fraud_audio")

# Create SQLite Database for PostgreSQL simulation
conn = sqlite3.connect("fraud_metadata.db", check_same_thread=False)
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

# -------------------------------------------------------------
# 2. FRONTEND UI (Same as Phase 3)
# -------------------------------------------------------------
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Clone Detection Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #1e1e2f; color: white; text-align: center; padding: 50px; }
        .container { max-width: 800px; margin: auto; background: #2a2a40; padding: 30px; border-radius: 15px; }
        h1 { color: #00d2ff; }
        .status { font-size: 20px; color: #a0a0b5; margin-bottom: 20px; }
        #alert-box { display: none; background: #ff4757; color: white; padding: 25px; border-radius: 10px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Smart India Hackathon '26</h1>
        <div class="status" id="connection-status">🟡 Connecting to Live Call...</div>
        
        <div id="alert-box">
            <h2>⚠️ CRITICAL ALERT SAVED TO DATABASE</h2>
            <div id="risk-score">Risk Score: 0%</div>
            <div id="keyword-match">Trigger Word: None</div>
        </div>
    </div>
    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const statusDiv = document.getElementById("connection-status");
        const alertBox = document.getElementById("alert-box");
        ws.onopen = () => { statusDiv.innerHTML = "🟢 Monitoring Call Live"; statusDiv.style.color = "#2ed573"; };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            alertBox.style.display = "block";
            document.getElementById("risk-score").innerHTML = "Risk Score: " + data.risk_score + "%";
            document.getElementById("keyword-match").innerHTML = "Trigger Word: " + data.keyword;
            statusDiv.innerHTML = "🔴 DATA SAVED FOR ACTIVE LEARNING";
            statusDiv.style.color = "#ff4757";
        };
    </script>
</body>
</html>
"""

# -------------------------------------------------------------
# 3. AI LOGIC & STORAGE (Phase 4 Additions)
# -------------------------------------------------------------
def run_l2_keyword_trigger():
    dummy_keywords = ["TRANSFER", "OTP", "FUND", "APPROVAL"]
    detected_word = np.random.choice(["HELLO", "TRANSFER", "OTP", "FUND", "OKAY"], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    if detected_word in dummy_keywords:
        return True, detected_word
    return False, None

def run_l3_heavy_ai():
    return round(np.random.uniform(76.0, 99.9), 2)

def model_worker_thread():
    global STOP_PROCESS
    silent_chunks_count = 0
    print("[Audio Engine] Started monitoring...")

    while not STOP_PROCESS:
        try: audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty: continue

        rms_energy = np.sqrt(np.mean(audio_chunk**2))

        if rms_energy < SILENCE_THRESHOLD:
            silent_chunks_count += 1
            if silent_chunks_count >= SILENT_CHUNKS_LIMIT:
                STOP_PROCESS = True
                break
        else:
            silent_chunks_count = 0
            l2_triggered, matched_word = run_l2_keyword_trigger()
            
            if l2_triggered:
                risk_score = run_l3_heavy_ai()
                
                if risk_score > 75:
                    print(f"🚨 ALERT! Risk: {risk_score}% | Saving Data for Active Learning...")
                    
                    # 1. Save Audio (Simulate S3)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_filename = f"fraud_audio/alert_{timestamp}.wav"
                    wavfile.write(audio_filename, SAMPLE_RATE, audio_chunk)
                    
                    # 2. Save Metadata (Simulate PostgreSQL)
                    cursor.execute('''
                        INSERT INTO alerts (timestamp, keyword, risk_score, audio_path)
                        VALUES (?, ?, ?, ?)
                    ''', (timestamp, matched_word, risk_score, audio_filename))
                    conn.commit()
                    
                    # 3. Trigger UI Alert
                    alert_queue.put({"type": "ALERT", "keyword": matched_word, "risk_score": risk_score})

        audio_queue.task_done()

def audio_callback(indata, frames, time_info, status):
    global frames_buffer
    if not STOP_PROCESS:
        frames_buffer.append(indata.copy())
        if len(frames_buffer) >= 50:
            audio_queue.put(np.concatenate(frames_buffer, axis=0))
            frames_buffer = []

# -------------------------------------------------------------
# 4. FASTAPI SERVER
# -------------------------------------------------------------
app = FastAPI()
connected_clients = []

@app.get("/")
async def get_dashboard(): return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            if not alert_queue.empty():
                await websocket.send_text(json.dumps(alert_queue.get()))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=model_worker_thread, daemon=True).start()
    app.state.stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=SAMPLES_PER_CHUNK, channels=1, callback=audio_callback)
    app.state.stream.start()

@app.on_event("shutdown")
async def shutdown_event():
    global STOP_PROCESS
    STOP_PROCESS = True
    app.state.stream.stop()
    app.state.stream.close()
    conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")