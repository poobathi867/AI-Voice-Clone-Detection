import queue
import threading
import time
import numpy as np
import sounddevice as sd

# -------------------------------------------------------------
# 1. SETTINGS & CONFIGURATION
# -------------------------------------------------------------
SAMPLE_RATE = 16000  # 16kHz (Standard for speech & AI models)
CHUNK_MS = 20  # 20ms raw audio frame
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_MS / 1000))  # 320 samples per 20ms

audio_queue = queue.Queue()
frames_buffer = []

# --- SMART AUTO-STOP CONFIGURATION ---
STOP_PROCESS = False
SILENCE_THRESHOLD = 0.001  # Energy below this is considered Silence
SILENT_CHUNKS_LIMIT = 2    # Stop after 2 continuous seconds of silence

# -------------------------------------------------------------
# 2. AUDIO CAPTURE CALLBACK
# -------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    global frames_buffer, STOP_PROCESS

    if STOP_PROCESS:
        return

    audio_frame = indata.copy()
    frames_buffer.append(audio_frame)

    # 50 frames = 1 Second Chunk
    if len(frames_buffer) >= 50:
        one_sec_chunk = np.concatenate(frames_buffer, axis=0)
        audio_queue.put(one_sec_chunk)
        frames_buffer = []


# -------------------------------------------------------------
# 3. WORKER THREAD (Smart Silence Detection)
# -------------------------------------------------------------
def model_worker_thread():
    global STOP_PROCESS
    print("[Worker Thread] Started & Listening to Queue...")

    chunk_count = 0
    silent_chunks_count = 0  # Counter for continuous silence

    while not STOP_PROCESS:
        try:
            audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        chunk_count += 1
        
        # Calculate Audio Energy (Volume)
        rms_energy = np.sqrt(np.mean(audio_chunk**2))
        
        print(f"\n [Queue Success] Pulled 1-Sec Chunk #{chunk_count} | Energy: {rms_energy:.6f}")

        # -------------------------------------------------------
        # SMART SILENCE DETECTION LOGIC
        # -------------------------------------------------------
        if rms_energy < SILENCE_THRESHOLD:
            # Silence detected
            silent_chunks_count += 1
            print(f"   -> 🤫 Silence detected... ({silent_chunks_count}/{SILENT_CHUNKS_LIMIT} seconds)")
            
            if silent_chunks_count >= SILENT_CHUNKS_LIMIT:
                print("\n 🎯 [SILENCE TIMEOUT] User finished speaking. Triggering Auto-Stop...")
                STOP_PROCESS = True
                audio_queue.task_done()
                break
        else:
            # Active speech detected, reset silence counter!
            silent_chunks_count = 0
            print("   -> 🗣️ Active Speech Detected. Processing...")
            # (Phase 2 L1/L2/L3 logic will be called here)

        audio_queue.task_done()

    print("\n🛑 [Worker Thread] Process Halted cleanly.")


# -------------------------------------------------------------
# 4. MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=== Starting Prototype Phase 1: Smart VAD Auto-Stop ===")

    worker = threading.Thread(target=model_worker_thread, daemon=True)
    worker.start()

    print("\n[Microphone] Intercepting audio stream (20ms frames)...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLES_PER_CHUNK,
        channels=1,
        callback=audio_callback,
    ):
        print("\nSpeak into mic. Prototype will AUTO-STOP only if you are SILENT for 2 seconds.\n")
        try:
            while not STOP_PROCESS:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
            
    print("\n=== Prototype Stopped successfully ===")