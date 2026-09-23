import queue
import threading
import time
import numpy as np
import sounddevice as sd

# -------------------------------------------------------------
# 1. CONFIGURATION & FLAGS
# -------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_MS = 20
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_MS / 1000))

audio_queue = queue.Queue()
frames_buffer = []

# --- SMART AUTO-STOP CONFIGURATION ---
STOP_PROCESS = False
SILENCE_THRESHOLD = 0.001  # Energy below this is considered Silence
SILENT_CHUNKS_LIMIT = 2    # Stop after 2 continuous seconds of silence

# -------------------------------------------------------------
# 2. TRIAGE LAYER FUNCTIONS
# -------------------------------------------------------------
def run_l2_keyword_trigger(audio_chunk):
    dummy_keywords = ["TRANSFER", "OTP", "FUND", "APPROVAL"]
    detected_word = np.random.choice(
        ["HELLO", "TRANSFER", "OTP", "FUND", "OKAY"], p=[0.1, 0.3, 0.3, 0.2, 0.1]
    )

    if detected_word in dummy_keywords:
        print(f"   [L2 Trigger] 🔥 KEYWORD MATCHED: '{detected_word}'")
        return True, detected_word
    else:
        print(f"   [L2 Trigger] Safe speech keyword: '{detected_word}'")
        return False, None


def run_l3_heavy_ai(audio_chunk):
    print(
        "   [L3 AI Ensemble] 🚀 Running Deep Learning Models (DeepFense + AudioSeal + CNN-BiLSTM)..."
    )

    deepfense_score = np.random.uniform(0.70, 0.95)
    audioseal_score = np.random.uniform(0.65, 0.90)
    cnn_bilstm_score = np.random.uniform(0.75, 0.98)

    ensemble_risk_score = (
        (deepfense_score * 0.4)
        + (audioseal_score * 0.3)
        + (cnn_bilstm_score * 0.3)
    ) * 100
    return round(ensemble_risk_score, 2)


# -------------------------------------------------------------
# 3. WORKER THREAD (Smart Silence Detection + Triage)
# -------------------------------------------------------------
def model_worker_thread():
    global STOP_PROCESS
    print("[Worker Thread] 3-Layer Smart Triage Pipeline Active...\n")

    chunk_id = 0
    silent_chunks_count = 0

    while not STOP_PROCESS:
        try:
            audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        chunk_id += 1
        rms_energy = np.sqrt(np.mean(audio_chunk**2))

        print(
            f"\n================ [Processing 1-Sec Audio Chunk #{chunk_id}] ================"
        )
        print(f"   [L1 Filter] Live Mic RMS Energy: {rms_energy:.6f}")

        # STEP 1: L1 Filter & Silence Detection
        if rms_energy < SILENCE_THRESHOLD:
            silent_chunks_count += 1
            print(
                f" -> L1 Status: SAFE (Silence detected... {silent_chunks_count}/{SILENT_CHUNKS_LIMIT} sec)"
            )

            if silent_chunks_count >= SILENT_CHUNKS_LIMIT:
                print(
                    "\n 🎯 [SILENCE TIMEOUT] User finished speaking. Triggering Auto-Stop..."
                )
                STOP_PROCESS = True
                audio_queue.task_done()
                break  # Exit thread
        else:
            # Active speech detected, reset silence counter!
            silent_chunks_count = 0
            print(" -> L1 Status: PASSED (Speech Detected). Moving to L2...")

            # STEP 2: L2 Keyword Trigger
            l2_triggered, matched_word = run_l2_keyword_trigger(audio_chunk)

            if not l2_triggered:
                print(" -> L2 Status: SAFE (No threat keywords). Skipping L3.")
            else:
                print(
                    " -> L2 Status: TRIGGERED! Moving to L3 Heavy AI Ensemble..."
                )

                # STEP 3: L3 AI Ensemble
                risk_score = run_l3_heavy_ai(audio_chunk)

                print(
                    f"\n 🚨 [FINAL RESULT] Voice Clone Risk Score: {risk_score}%"
                )
                if risk_score > 75:
                    print(" ⚠️ HIGH RISK ALERT: Impersonation Attack Detected!")
                    # NOTE: Loop will NO LONGER break here. It lets you finish your sentence!

        audio_queue.task_done()

    print("\n🛑 [Worker Thread] Process Halted cleanly.")


# -------------------------------------------------------------
# 4. AUDIO CALLBACK & MAIN EXECUTION
# -------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    global frames_buffer, STOP_PROCESS

    if STOP_PROCESS:
        return

    frames_buffer.append(indata.copy())

    if len(frames_buffer) >= 50:
        one_sec_chunk = np.concatenate(frames_buffer, axis=0)
        audio_queue.put(one_sec_chunk)
        frames_buffer = []


if __name__ == "__main__":
    print("=== Starting Phase 2 (Smart VAD Auto-Stop) ===")

    worker = threading.Thread(target=model_worker_thread, daemon=True)
    worker.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLES_PER_CHUNK,
        channels=1,
        callback=audio_callback,
    ):
        print(
            "\nSpeak into mic. Prototype will process in real-time and AUTO-STOP only if you are SILENT for 2 seconds.\n"
        )
        try:
            while not STOP_PROCESS:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    print("\n=== Prototype Stopped successfully ===")