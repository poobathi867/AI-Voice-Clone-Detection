import queue
import threading
import time
import numpy as np
import sounddevice as sd
import asyncio
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# -------------------------------------------------------------
# 1. CONFIGURATION & FLAGS
# -------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_MS = 20
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_MS / 1000))

audio_queue = queue.Queue()
frames_buffer = []

STOP_PROCESS = False
SILENCE_THRESHOLD = 0.001
SILENT_CHUNKS_LIMIT = 2

# Queue to send alerts to the WebSocket clients
alert_queue = queue.Queue()

# -------------------------------------------------------------
# 2. FRONTEND UI (HTML + JavaScript + WebSockets)
# -------------------------------------------------------------
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Voice Clone Detection Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e2f; color: white; text-align: center; padding: 50px; }
        .container { max-width: 800px; margin: auto; background: #2a2a40; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        h1 { color: #00d2ff; }
        .status { font-size: 20px; color: #a0a0b5; margin-bottom: 20px; }
        #alert-box { display: none; background: #ff4757; color: white; padding: 25px; border-radius: 10px; margin-top: 30px; border: 3px solid #ff0000; animation: blink 1s infinite; }
        #alert-box h2 { margin: 0; font-size: 35px; text-transform: uppercase; }
        #risk-score { font-size: 28px; font-weight: bold; margin-top: 10px; }
        #keyword-match { font-size: 20px; margin-top: 5px; font-style: italic; }
        @keyframes blink { 0% { box-shadow: 0 0 10px #ff4757; } 50% { box-shadow: 0 0 40px #ff4757; } 100% { box-shadow: 0 0 10px #ff4757; } }
        .footer { margin-top: 40px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Smart India Hackathon '26</h1>
        <h2>AI Voice Clone Prevention System</h2>
        <div class="status" id="connection-status">🟡 Connecting to Live Call...</div>
        
        <div id="alert-box">
            <h2>⚠️ CRITICAL ALERT</h2>
            <div>Impersonation Attack Detected in Live Call!</div>
            <div id="risk-score">Risk Score: 0%</div>
            <div id="keyword-match">Trigger Word: None</div>
        </div>
        
        <div class="footer">Phase 3 Prototype: Real-Time WebSocket Alerts</div>
    </div>

    <script>
        // Connect to FastAPI WebSocket
        const ws = new WebSocket("ws://localhost:8000/ws");
        const statusDiv = document.getElementById("connection-status");
        const alertBox = document.getElementById("alert-box");
        const scoreDiv = document.getElementById("risk-score");
        const keywordDiv = document.getElementById("keyword-match");

        ws.onopen = function() {
            statusDiv.innerHTML = "🟢 System Active - Monitoring Call Live";
            statusDiv.style.color = "#2ed573";
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if(data.type === "ALERT") {
                // Show the alert box with data
                alertBox.style.display = "block";
                scoreDiv.innerHTML = "Risk Score: " + data.risk_score + "%";
                keywordDiv.innerHTML = "Trigger Word: " + data.keyword;
                statusDiv.innerHTML = "🔴 CALL COMPROMISED";
                statusDiv.style.color = "#ff4757";
            }
        };

        ws.onclose = function() {
            statusDiv.innerHTML = "⚪ Call Ended / System Offline";
            statusDiv.style.color = "#a0a0b5";
        };
    </script>
</body>
</html>
"""

# -------------------------------------------------------------
# 3. TRIAGE AI LOGIC (From Phase 2)
# -------------------------------------------------------------
def run_l2_keyword_trigger():
    dummy_keywords = ["TRANSFER", "OTP", "FUND", "APPROVAL"]
    detected_word = np.random.choice(["HELLO", "TRANSFER", "OTP", "FUND", "OKAY"], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    if detected_word in dummy_keywords:
        return True, detected_word
    return False, None

def run_l3_heavy_ai():
    deepfense_score = np.random.uniform(0.70, 0.95)
    audioseal_score = np.random.uniform(0.65, 0.90)
    cnn_bilstm_score = np.random.uniform(0.75, 0.98)
    return round(((deepfense_score * 0.4) + (audioseal_score * 0.3) + (cnn_bilstm_score * 0.3)) * 100, 2)

# -------------------------------------------------------------
# 4. AUDIO PIPELINE THREAD
# -------------------------------------------------------------
def model_worker_thread():
    global STOP_PROCESS
    silent_chunks_count = 0
    print("[Audio Engine] Started monitoring...")

    while not STOP_PROCESS:
        try:
            audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        rms_energy = np.sqrt(np.mean(audio_chunk**2))

        if rms_energy < SILENCE_THRESHOLD:
            silent_chunks_count += 1
            if silent_chunks_count >= SILENT_CHUNKS_LIMIT:
                print(" -> User finished speaking. Stopping audio pipeline.")
                STOP_PROCESS = True
                break
        else:
            silent_chunks_count = 0
            l2_triggered, matched_word = run_l2_keyword_trigger()
            
            if l2_triggered:
                risk_score = run_l3_heavy_ai()
                print(f"🚨 ALERT GENERATED: {matched_word} | Risk: {risk_score}%")
                
                # Push the alert data to the WebSocket queue
                if risk_score > 75:
                    alert_queue.put({
                        "type": "ALERT",
                        "keyword": matched_word,
                        "risk_score": risk_score
                    })

        audio_queue.task_done()

def audio_callback(indata, frames, time_info, status):
    global frames_buffer, STOP_PROCESS
    if STOP_PROCESS: return
    frames_buffer.append(indata.copy())
    if len(frames_buffer) >= 50:
        audio_queue.put(np.concatenate(frames_buffer, axis=0))
        frames_buffer = []

# -------------------------------------------------------------
# 5. FASTAPI & WEBSOCKET SERVER
# -------------------------------------------------------------
app = FastAPI()
connected_clients = []

@app.get("/")
async def get_dashboard():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Check if there are any alerts in the queue from the Audio Thread
            if not alert_queue.empty():
                alert_data = alert_queue.get()
                # Send the alert JSON to the browser
                await websocket.send_text(json.dumps(alert_data))
                alert_queue.task_done()
            await asyncio.sleep(0.1) # Small sleep to prevent blocking
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.on_event("startup")
async def startup_event():
    # Start the audio interception when the server starts
    threading.Thread(target=model_worker_thread, daemon=True).start()
    
    # Keep reference to the InputStream so it stays active
    app.state.stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLES_PER_CHUNK,
        channels=1,
        callback=audio_callback
    )
    app.state.stream.start()

@app.on_event("shutdown")
async def shutdown_event():
    global STOP_PROCESS
    STOP_PROCESS = True
    app.state.stream.stop()
    app.state.stream.close()

if __name__ == "__main__":
    print("\n🚀 Starting FastApi Server for Phase 3...")
    print("👉 Open your browser and go to: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")