from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import numpy as np

from voice_features import extract_features_from_bytes


app = FastAPI(title="NeuroSense Voice ML API")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "voice_model.joblib"


# Load the voice model once when the service starts.
print("[VOICE] Loading voice model...", flush=True)

try:
    model_bundle = joblib.load(MODEL_PATH)
    voice_model = model_bundle["pipeline"]
    print("[VOICE] Voice model loaded successfully.", flush=True)
except Exception as e:
    voice_model = None
    print(f"[VOICE] Model loading failed: {e}", flush=True)


@app.get("/")
def root():
    return {
        "service": "NeuroSense Voice ML",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": voice_model is not None,
    }


@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...)):

    if voice_model is None:
        raise HTTPException(
            status_code=503,
            detail="Voice model is not loaded",
        )

    filename = file.filename or "audio.webm"
    suffix = Path(filename).suffix.lower()

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".webm",
    }

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}",
        )

    try:
        print(
            f"[VOICE] Received {filename}",
            flush=True,
        )

        audio_bytes = await file.read()

        print(
            f"[VOICE] Audio size: {len(audio_bytes)} bytes",
            flush=True,
        )

        print(
            "[VOICE] Starting feature extraction...",
            flush=True,
        )

        features = extract_features_from_bytes(
            audio_bytes,
            suffix,
        )

        print(
            f"[VOICE] Extracted {len(features)} features",
            flush=True,
        )

        if len(features) != 78:
            raise ValueError(
                f"Expected 78 features, got {len(features)}"
            )

        X = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, -1)

        print(
            "[VOICE] Starting prediction...",
            flush=True,
        )

        prediction = voice_model.predict(X)[0]

        print(
            f"[VOICE] Prediction: {prediction}",
            flush=True,
        )

        probability = None

        if hasattr(voice_model, "predict_proba"):
            probabilities = voice_model.predict_proba(X)[0]

            classes = list(voice_model.classes_)

            probability_map = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities)
            }

            probability = float(
                max(probabilities)
            )

            print(
                f"[VOICE] Probabilities: {probability_map}",
                flush=True,
            )

        prediction = str(prediction).lower()

        if prediction == "1":
            result = "elevated"
        else:
            result = "low"

        return {
            "prediction": prediction,
            "probability": probability,
            "result": result,
            "disclaimer": (
                "This result is for screening purposes only "
                "and is not a medical diagnosis."
            ),
        }

    except Exception as e:

        print(
            f"[VOICE ERROR] {type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Voice prediction failed: {str(e)}",
        )