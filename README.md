Here are the tools and libraries used in your code, with a short explanation of each.

## Core libraries:

1. Streamlit (`streamlit as st`)
The web app framework. It turns your Python script into an interactive web page without any HTML/JS. You use it for:
st.set_page_config`, `st.title`: page title and icon
st.sidebar.selectbox`: model and language dropdowns
st.tabs`: the Record and Upload tabs
st.audio_input`: records audio from the microphone
st.file_uploader`: uploads audio files
st.button`, `st.spinner`, `st.caption`, `st.text_area`, `st.download_button`: UI controls and output
st.session_state`: stores the transcript so it persists between reruns
@st.cache_resource`: loads the Whisper model once and reuses it

2. faster-whisper (`WhisperModel`)
The speech-to-text engine. It is a faster re-implementation of OpenAI's Whisper model using CTranslate2. It:
- Downloads and runs the model locally (no API or internet needed after the first download)
- Supports multiple sizes (tiny to large-v3) that trade speed for accuracy
- Uses `compute_type="int8"` to reduce memory and speed up CPU inference
- Provides `vad_filter=True`, which skips silent parts, and `beam_size=5` for better decoding accuracy

3. NumPy (`numpy as np`)
Used for numerical audio processing in `wav_to_array`:
- Converts raw bytes to arrays (`np.frombuffer`)
- Normalizes samples to the float range -1 to 1
- Averages stereo channels into mono
- Resamples to 16 kHz with `np.interp` and `np.linspace`

## Standard library modules

4. `wave`
Reads WAV files and extracts the sample rate, channel count, sample width, and raw audio frames. This lets you decode microphone recordings without ffmpeg or PyAV.

5. `io` (io.BytesIO) 
Wraps the in-memory audio bytes so `wave` can read them like a file, with no disk write.

6. `tempfile`
Creates a temporary file for uploaded formats like mp3 or m4a, since the decoder needs a file path.

7. `os`
Used for file-system tasks: `os.path.splitext` gets the file extension, and `os.remove` deletes the temp file after transcription.

Behind the scenes (not imported directly)

PyAV / ffmpeg libraries
`faster-whisper` uses these internally to decode non-WAV formats (mp3, m4a, ogg, etc.). Your code avoids them for WAV, but they are needed for other formats.

## Summary:

|     Tool    |         |   Role                   |
| Streamlit             | UI / web app             |
| faster-whisper        | Speech recognition       |
| NumPy                 | Audio array processing   |
| wave + io             | Decode WAV in memory     |
| tempfile + os         | Handle uploaded files    |
