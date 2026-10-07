import io
import os
import tempfile
import wave

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Speech to Text", page_icon="🎙️")
st.title("🎙️ Speech to Text (runs locally)")

# ---------- Sidebar settings ----------
model_size = st.sidebar.selectbox(
    "Model size",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=1,
    help="Bigger = more accurate but slower and needs more RAM.",
)
language = st.sidebar.selectbox(
    "Language",
    ["auto-detect", "en", "hi", "pa", "ur", "es", "fr", "de", "ar", "zh", "ja"],
    index=0,
)


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(size: str) -> WhisperModel:
    # CPU-friendly settings. If you have an NVIDIA GPU, use device="cuda", compute_type="float16"
    return WhisperModel(size, device="cpu", compute_type="int8")


def wav_to_array(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode WAV bytes to mono float32 at 16 kHz without needing PyAV/ffmpeg."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        data = (data - 128) / 128.0
    else:
        data /= float(np.iinfo(dtype).max)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        n = int(len(data) * target_sr / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return data.astype(np.float32)


def transcribe(audio_bytes: bytes, suffix: str = ".wav") -> str:
    model = load_model(model_size)
    lang = None if language == "auto-detect" else language
    opts = dict(language=lang, vad_filter=True, beam_size=5)

    if suffix.lower() == ".wav":
        # Plain WAV (e.g. from the microphone): decode ourselves, no PyAV needed.
        segments, info = model.transcribe(wav_to_array(audio_bytes), **opts)
        text = " ".join(seg.text.strip() for seg in segments)
    else:
        # Other formats (mp3, m4a, ...) need PyAV to decode.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            segments, info = model.transcribe(tmp_path, **opts)
            text = " ".join(seg.text.strip() for seg in segments)
        finally:
            os.remove(tmp_path)

    st.caption(f"Detected language: {info.language} ({info.language_probability:.0%})")
    return text


tab_mic, tab_file = st.tabs(["🎤 Record", "📁 Upload file"])

with tab_mic:
    audio = st.audio_input("Click to record, click again to stop")
    if audio is not None:
        if st.button("Transcribe recording", type="primary"):
            with st.spinner("Transcribing..."):
                result = transcribe(audio.getvalue(), ".wav")
            st.session_state["result"] = result

with tab_file:
    uploaded = st.file_uploader(
        "Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"]
    )
    if uploaded is not None:
        st.audio(uploaded)
        if st.button("Transcribe file", type="primary"):
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            with st.spinner("Transcribing..."):
                result = transcribe(uploaded.getvalue(), suffix)
            st.session_state["result"] = result

if st.session_state.get("result"):
    st.subheader("Transcript")
    text = st.text_area("Result", st.session_state["result"], height=250)
    st.download_button("Download as .txt", text, file_name="transcript.txt")
