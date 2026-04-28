"""
ISL Recognition Backend — FastAPI + WebSocket (Optimised)
=========================================================
Architecture change: MediaPipe runs IN THE BROWSER (JS SDK).
The client extracts 126-D hand landmarks and sends ONLY those numbers.
This backend does nothing but:
  1. Receive a tiny JSON payload (~1 KB vs ~50 KB JPEG frame)
  2. Run the Keras model (~1 ms)
  3. Return the prediction

Result: end-to-end latency drops from 200–500 ms → < 30 ms.

Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Labels ────────────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "label_classes.json"), "r") as f:
    LABELS: list[str] = json.load(f)
NUM_CLASSES = len(LABELS)
print(f"[ISL] Labels loaded ({NUM_CLASSES}): {LABELS}")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> Sequential:
    m = Sequential([
        Dense(256, activation="relu", input_shape=(126,)),
        BatchNormalization(), Dropout(0.4),
        Dense(256, activation="relu"),
        BatchNormalization(), Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(), Dropout(0.3),
        Dense(64,  activation="relu"),
        BatchNormalization(), Dropout(0.2),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

isl_model = build_model()
isl_model.load_weights(os.path.join(BASE_DIR, "isl_mediapipe.weights.h5"))

# Warm-up: first inference is always slow due to TF graph compilation.
_ = isl_model.predict(np.zeros((1, 126)), verbose=0)
print("[ISL] Model ready (warmed up).")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="ISL Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from ./static/
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "labels": NUM_CLASSES}


# ── Per-session state ──────────────────────────────────────────────────────────
class SessionState:
    """Holds per-connection buffers and sentence state."""

    CONFIDENCE_THRESHOLD = 0.50
    HOLD_SECONDS         = 0.8
    NO_HAND_TIMEOUT      = 2.0
    SMOOTH_WINDOW        = 5        # frames to average predictions over

    def __init__(self):
        self.pred_buffer: deque[np.ndarray] = deque(maxlen=self.SMOOTH_WINDOW)
        self.sentence: list[str] = []
        self.current_word   = ""
        self.current_letter: str | None = None
        self.letter_hold_start: float | None = None
        self.last_hand_time = time.monotonic()

    # ── Helpers ───────────────────────────────────────────────
    def clear(self):
        self.sentence.clear()
        self.current_word = ""
        self.pred_buffer.clear()
        self.current_letter = None
        self.letter_hold_start = None

    def add_space(self):
        self.sentence.append(" ")
        self.current_word = ""

    def backspace(self):
        if self.sentence:
            self.sentence.pop()
            # keep current_word in sync
            self.current_word = "".join(
                c for c in self.sentence if c != " "
            ).split()[-1] if self.sentence else ""

    @property
    def full_sentence(self) -> str:
        return "".join(self.sentence)


# ── Prediction logic ───────────────────────────────────────────────────────────
def run_inference(coords: list[float], state: SessionState) -> dict:
    """
    coords: 126 floats already normalised by the JS client.
    Returns a dict with prediction results and updated sentence info.
    """
    arr  = np.array(coords, dtype=np.float32).reshape(1, 126)
    pred = isl_model.predict(arr, verbose=0)[0]
    state.pred_buffer.append(pred)

    avg      = np.mean(state.pred_buffer, axis=0)
    top3_idx = np.argsort(avg)[::-1][:3]
    top3     = [{"label": LABELS[i].upper(), "conf": round(float(avg[i]), 3)}
                for i in top3_idx]

    best_idx        = int(top3_idx[0])
    best_confidence = float(avg[best_idx])
    best_label      = LABELS[best_idx] if best_confidence >= state.CONFIDENCE_THRESHOLD else "..."

    letter_recorded: str | None = None
    now = time.monotonic()
    state.last_hand_time = now

    if best_label != "...":
        if best_label == state.current_letter:
            held = now - (state.letter_hold_start or now)
            if held >= state.HOLD_SECONDS:
                letter_recorded = best_label.upper()
                state.current_word += letter_recorded
                state.sentence.append(letter_recorded)
                state.current_letter    = None
                state.letter_hold_start = None
        else:
            state.current_letter    = best_label
            state.letter_hold_start = now
    else:
        state.current_letter    = None
        state.letter_hold_start = None

    return {
        "best_label":      best_label.upper(),
        "best_confidence": round(best_confidence, 3),
        "top3":            top3,
        "sentence":        state.full_sentence,
        "current_word":    state.current_word,
        "letter_recorded": letter_recorded,
    }


def check_no_hand_timeout(state: SessionState):
    """Auto-space after hand disappears for NO_HAND_TIMEOUT seconds."""
    if (
        time.monotonic() - state.last_hand_time > state.NO_HAND_TIMEOUT
        and state.current_word
    ):
        state.add_space()
        state.last_hand_time = time.monotonic()


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = SessionState()
    print("[ISL] Client connected.")

    try:
        while True:
            data: dict = await ws.receive_json()
            action = data.get("action", "")

            # ── Control actions ──────────────────────────────
            if action == "clear":
                state.clear()
                await ws.send_json({"type": "cleared",
                                    "sentence": "", "current_word": ""})
                continue

            if action == "space":
                state.add_space()
                await ws.send_json({"type": "space",
                                    "sentence": state.full_sentence,
                                    "current_word": state.current_word})
                continue

            if action == "backspace":
                state.backspace()
                await ws.send_json({"type": "backspace",
                                    "sentence": state.full_sentence,
                                    "current_word": state.current_word})
                continue

            # ── No-hand timeout check ────────────────────────
            if action == "no_hand":
                check_no_hand_timeout(state)
                state.pred_buffer.clear()
                state.current_letter    = None
                state.letter_hold_start = None
                await ws.send_json({"type": "no_hand",
                                    "sentence": state.full_sentence,
                                    "current_word": state.current_word})
                continue

            # ── Landmark inference ───────────────────────────
            coords = data.get("coords")          # list of 126 floats
            if not coords or len(coords) != 126:
                continue

            result = run_inference(coords, state)
            await ws.send_json({"type": "prediction", **result})

    except WebSocketDisconnect:
        print("[ISL] Client disconnected.")
    except Exception as exc:
        print(f"[ISL] Error: {exc}")
        try:
            await ws.close()
        except Exception:
            pass
