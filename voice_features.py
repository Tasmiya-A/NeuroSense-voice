from __future__ import annotations

import io
import subprocess

import imageio_ffmpeg
import librosa
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

FEATURE_COUNT = 78

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
}


# ============================================================
# SAFE STATISTICS
# ============================================================

def mean_std(values):
    values = np.asarray(values)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return [0.0, 0.0]

    return [
        float(np.mean(values)),
        float(np.std(values)),
    ]


# ============================================================
# PITCH
# ============================================================

def extract_pitch(y, sr):
    try:
        f0 = librosa.yin(
            y,
            fmin=50,
            fmax=min(500, sr / 2 - 1),
            sr=sr,
        )

        f0 = f0[np.isfinite(f0)]

        f0 = f0[
            (f0 >= 50)
            & (f0 <= 500)
        ]

        return f0

    except Exception:
        return np.array([])


# ============================================================
# JITTER
# ============================================================

def calculate_jitter(f0):
    if len(f0) < 3:
        return 0.0

    periods = 1.0 / f0

    differences = np.abs(
        np.diff(periods)
    )

    mean_period = np.mean(periods)

    if mean_period <= 0:
        return 0.0

    return float(
        np.mean(differences)
        / mean_period
    )


# ============================================================
# SHIMMER
# ============================================================

def calculate_shimmer(rms):
    if len(rms) < 3:
        return 0.0

    rms = rms[
        np.isfinite(rms)
    ]

    if len(rms) < 3:
        return 0.0

    differences = np.abs(
        np.diff(rms)
    )

    mean_amplitude = np.mean(rms)

    if mean_amplitude <= 0:
        return 0.0

    return float(
        np.mean(differences)
        / mean_amplitude
    )


# ============================================================
# HNR
# ============================================================

def calculate_hnr(y):
    try:
        harmonic, percussive = (
            librosa.effects.hpss(y)
        )

        harmonic_energy = np.mean(
            harmonic ** 2
        )

        noise_energy = np.mean(
            percussive ** 2
        )

        if noise_energy <= 1e-10:
            return 40.0

        if harmonic_energy <= 1e-10:
            return 0.0

        hnr = 10 * np.log10(
            harmonic_energy
            / noise_energy
        )

        return float(
            np.clip(hnr, -20, 40)
        )

    except Exception:
        return 0.0


# ============================================================
# FEATURE VECTOR
# ============================================================

def feature_vector(y, sr):
    # Keep audio small and predictable in memory
    y, _ = librosa.effects.trim(y, top_db=30)

    if y.size < max(512, sr // 10):
        raise ValueError("audio is too short after trimming")

    # Use float32 throughout
    y = np.asarray(y, dtype=np.float32)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    print("[VOICE DEBUG] Starting MFCC...", flush=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=10,
        dtype=np.float32,
    )

    print("[VOICE DEBUG] MFCC done", flush=True)

    print("[VOICE DEBUG] Starting delta...", flush=True)

    delta = librosa.feature.delta(mfcc)

    print("[VOICE DEBUG] Delta done", flush=True)

    print("[VOICE DEBUG] Starting delta2...", flush=True)

    delta2 = librosa.feature.delta(mfcc, order=2)

    print("[VOICE DEBUG] Delta2 done", flush=True)

    print("[VOICE DEBUG] Starting pitch/YIN...", flush=True)

    f0 = extract_pitch(y, sr)

    print("[VOICE DEBUG] Pitch/YIN done", flush=True)

    pitch_mean, pitch_std = mean_std(f0)

    # Calculate RMS once and reuse it
    rms = librosa.feature.rms(y=y)

    jitter = calculate_jitter(f0)
    shimmer = calculate_shimmer(rms[0])

    print("[VOICE DEBUG] Starting HNR/HPSS...", flush=True)

    hnr = calculate_hnr(y)

    print("[VOICE DEBUG] HNR/HPSS done", flush=True)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)

    features = (
        list(mfcc.mean(axis=1))
        + list(mfcc.std(axis=1))

        + list(delta.mean(axis=1))
        + list(delta.std(axis=1))

        + list(delta2.mean(axis=1))
        + list(delta2.std(axis=1))

        + [pitch_mean, pitch_std]

        + [jitter, shimmer, hnr]

        + list(mean_std(zcr))
        + list(mean_std(rms))
        + list(mean_std(centroid))
        + list(mean_std(bandwidth))
        + list(mean_std(rolloff))
        + list(mean_std(flatness))

        + [len(y) / sr]
    )

    # Convert only once at the end
    result = np.asarray(features, dtype=np.float32)

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if result.shape[0] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} voice features, got {result.shape[0]}"
        )

    print("[VOICE DEBUG] All features calculated", flush=True)

    return result
# ============================================================
# DIRECT AUDIO-BYTES DECODER
# ============================================================

def decode_audio_bytes_to_array(
    audio_bytes: bytes,
    suffix: str,
    sr: int = SAMPLE_RATE,
):
    if not audio_bytes:
        raise ValueError(
            "audio data is empty"
        )

    print(
        "[VOICE DEBUG] Starting FFmpeg...",
        flush=True,
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    print(
        f"[VOICE DEBUG] FFmpeg executable: {ffmpeg}",
        flush=True,
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "pipe:1",
    ]

    print(
        "[VOICE DEBUG] Running FFmpeg...",
        flush=True,
    )

    process = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )

    print(
        f"[VOICE DEBUG] FFmpeg finished: "
        f"returncode={process.returncode}",
        flush=True,
    )

    if process.returncode != 0:
        error_message = process.stderr.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        raise RuntimeError(
            f"FFmpeg failed: "
            f"{error_message or 'unknown error'}"
        )

    if not process.stdout:
        raise ValueError(
            "FFmpeg produced no audio output"
        )

    audio = np.frombuffer(
        process.stdout,
        dtype=np.float32,
    ).copy()

    if audio.size == 0:
        raise ValueError(
            "Decoded audio contains no samples"
        )

    print(
        f"[VOICE DEBUG] FFmpeg decoded "
        f"{audio.size} samples",
        flush=True,
    )

    return audio, sr


# ============================================================
# FEATURES FROM BYTES
# ============================================================

def extract_features_from_bytes(
    audio_bytes: bytes,
    suffix: str,
    sr: int = SAMPLE_RATE,
):
    suffix = suffix.lower()

    if suffix not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {suffix}"
        )

    if suffix == ".wav":
        print(
            "[VOICE DEBUG] Loading WAV with librosa...",
            flush=True,
        )

        y, actual_sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=sr,
            mono=True,
        )

    else:
        print(
            f"[VOICE DEBUG] Decoding {suffix} "
            f"using FFmpeg...",
            flush=True,
        )

        y, actual_sr = decode_audio_bytes_to_array(
            audio_bytes,
            suffix,
            sr,
        )

    print(
        f"[VOICE DEBUG] Decoding complete: "
        f"samples={len(y)}, sr={actual_sr}",
        flush=True,
    )

    print(
        "[VOICE DEBUG] Starting librosa feature extraction...",
        flush=True,
    )

    features = feature_vector(
        y,
        actual_sr,
    )

    print(
        "[VOICE DEBUG] Librosa feature extraction complete.",
        flush=True,
    )

    return features


# ============================================================
# FEATURES FROM FILE
# ============================================================

def extract_features(path):
    """
    Extract the same 78 features from a training audio file.
    Training and API inference therefore use the same
    feature extraction implementation.
    """

    path = str(path)

    suffix = path.lower()

    if "." in suffix:
        suffix = "." + suffix.rsplit(".", 1)[1]

    with open(path, "rb") as f:
        audio_bytes = f.read()

    return extract_features_from_bytes(
        audio_bytes,
        suffix,
        SAMPLE_RATE,
    )