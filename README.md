# AI-Voice-Clone-Detection

## Technology Used:
- **Telephony:** PJSIP/OpenSIPS media fork, SIPS (TLS), SRTP, WebSockets
- **Detection models:** AASIST, CNN-BiLSTM, wav2vec 2.0 XLS-R (PyTorch)
- **Watermark check:** AudioSeal detector for watermarked TTS
- **Audio:** Librosa, SoundFile, noisereduce, WebRTC VAD, Mel-Spectrogram
- **Features:** log-mel, LFCC, MFCC, pitch, ZCR
- **Keyword trigger:** Vosk offline ASR ("OTP", "transfer", "urgent")
- **Serving:** ONNX Runtime, FastAPI, thread-safe queues
- **Storage:** PostgreSQL metadata; encrypted S3/IPFS for consented flagged clips
- **Security:** TLS 1.3, AES-256, RBAC, Immutable Blockchain Ledger (for tamper-proof metadata audit trail)
- **Training data:** ASVspoof 2019 LA / 2021 DF, In-the-Wild