import os
import uuid
import threading
from datetime import timedelta
from typing import Optional, List, Tuple
import subprocess

from flask import Flask, request, jsonify, render_template
import torch
import whisper
import librosa
import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)

###############################################
# DETECCIÓN DE GPU vs CPU
###############################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ GPU detectada. Usando CUDA." if DEVICE == "cuda" else "⚠️ No se detecta GPU. Usando CPU.")

###############################################
# CARGAR MODELO WHISPER
###############################################
print("Cargando modelo Whisper ('base') en", DEVICE, "...")
whisper_model = whisper.load_model("base", device=DEVICE)
print("✅ Modelo Whisper listo.")

# Constantes generales
DEFAULT_SR = 16000  # Sample‑rate por defecto

###############################################
# UTILIDADES
###############################################

def extract_audio(input_file: str, output_file: str, sr: int = DEFAULT_SR) -> None:
    """Extrae la pista de *input_file* a WAV mono *sr* Hz con ffmpeg."""
    cmd = [
        "ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1", output_file, "-y",
    ]
    subprocess.run(cmd, check=True)


###############################################
# MEMORIA DE TRABAJOS (job_id -> estado)
###############################################
JOBS: dict[str, dict] = {}


###############################################
# PIPELINE MODULAR
###############################################

def transcribe_audio(audio_path: str, language: Optional[str] = None):
    """Ejecuta Whisper y devuelve la lista de *segments*."""
    whisper_kwargs = {"verbose": False}
    if language and language.lower() != "auto":
        whisper_kwargs["language"] = language.lower()
    result = whisper_model.transcribe(audio_path, **whisper_kwargs)
    return result["segments"]


def create_embeddings(segments, wav: np.ndarray, sr: int) -> List[np.ndarray]:
    """Genera embeddings de voz para cada segmento."""
    encoder = VoiceEncoder()
    embeds = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        audio_slice = wav[start_sample:end_sample]
        if len(audio_slice) < sr * 0.5:  # < 0,5 s
            embeds.append(np.zeros(256))
        else:
            embeds.append(encoder.embed_utterance(audio_slice))
    return embeds


def choose_num_speakers(embeddings: List[np.ndarray]) -> int:
    """Elige *k* con mejor silhouette en el rango 2‑9."""
    X = np.vstack(embeddings)
    best_k, best_score = 2, -1
    for k in range(2, 10):
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_embeddings(embeddings: List[np.ndarray], n_speakers: int) -> np.ndarray:
    """Retorna las etiquetas de KMeans con *n_speakers* clusters."""
    X = np.vstack(embeddings)
    return KMeans(n_clusters=n_speakers, random_state=0).fit_predict(X)


def aggregate_segments(segments, labels) -> List[Tuple[float, float, str, str]]:
    """Agrupa segmentos contiguos del mismo locutor."""
    aggregated = []
    last_spk, text_acc = None, ""
    start_acc = end_acc = None
    for seg, lbl in zip(segments, labels):
        speaker = f"SPEAKER_{lbl}"
        text, start, end = seg["text"].strip(), seg["start"], seg["end"]
        if speaker == last_spk:
            text_acc += " " + text
            end_acc = end
        else:
            if last_spk is not None:
                aggregated.append((start_acc, end_acc, last_spk, text_acc.strip()))
            last_spk, text_acc = speaker, text
            start_acc, end_acc = start, end
    if text_acc:
        aggregated.append((start_acc, end_acc, last_spk, text_acc.strip()))
    return aggregated


def format_txt_result(aggregated_segments) -> str:
    """Devuelve una cadena multilínea legible para humanos."""
    lines = []
    for start, end, speaker, texto in aggregated_segments:
        start_time = str(timedelta(seconds=int(start)))
        end_time = str(timedelta(seconds=int(end)))
        lines.append(f"[{start_time} - {end_time}] {speaker}: {texto}")
    return "\n".join(lines)


def build_json_result(aggregated_segments, n_speakers: int, language: str):
    """Construye el JSON final."""
    return {
        "speakers_detected": n_speakers,
        "language": language or "auto",
        "segments": [
            {
                "start": float(start),
                "end": float(end),
                "speaker": speaker,
                "text": texto,
            }
            for start, end, speaker, texto in aggregated_segments
        ],
    }


###############################################
# FUNCIÓN ORQUESTADORA (ahora mucho más ligera)
###############################################

def transcribe_in_background(
    audio_path: str,
    job_id: str,
    n_speakers: Optional[int] = None,
    language: Optional[str] = None,
) -> None:
    job_info = JOBS[job_id]
    try:
        # 1️⃣ Transcripción
        _update_job(job_id, 5, "Audio cargado, iniciando transcripción…")
        _update_job(job_id, 20, "Transcribiendo…")
        segments = transcribe_audio(audio_path, language)

        # 2️⃣ Embeddings y clustering (si corresponde)
        if n_speakers == 1:
            _update_job(job_id, 50, "Un solo hablante. Saltando embeddings/clustering…")
            labels = np.zeros(len(segments), dtype=int)
        else:
            _update_job(job_id, 60, "Generando embeddings…")
            wav, sr = librosa.load(audio_path, sr=DEFAULT_SR)
            embeddings = create_embeddings(segments, wav, sr)

            if n_speakers is None:
                _update_job(job_id, 75, "Determinando número de locutores…")
                n_speakers = choose_num_speakers(embeddings)
            _update_job(job_id, 85, f"Clustering en {n_speakers} locutores…")
            labels = cluster_embeddings(embeddings, n_speakers)

        # 3️⃣ Agrupar y formatear resultados
        _update_job(job_id, 90, "Agrupando intervenciones…")
        aggregated = aggregate_segments(segments, labels)
        txt_result = format_txt_result(aggregated)
        json_result = build_json_result(aggregated, n_speakers or 1, language)

        # 4️⃣ Finalizar
        job_info.update(
            progress=100,
            status="Completado",
            completed=True,
            txt_result=txt_result,
            json_result=json_result,
        )
    except Exception as exc:
        job_info.update(
            progress=100,
            status=f"Error: {exc}",
            completed=True,
            txt_result=f"ERROR: {exc}",
            json_result={"error": str(exc)},
        )
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def _update_job(job_id: str, progress: int, status: str) -> None:
    job_info = JOBS[job_id]
    job_info["progress"] = progress
    job_info["status"] = status


###############################################
# ENDPOINTS FLASK (sin cambios funcionales)
###############################################
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return jsonify({"error": "No se envió ningún archivo 'audio_file'"}), 400

    n_speakers_param = request.form.get("n_speakers")
    n_speakers = int(n_speakers_param) if n_speakers_param and n_speakers_param.isdigit() else None
    language_param = request.form.get("language", "auto").strip().lower() or "auto"

    job_id = str(uuid.uuid4())
    temp_filename = f"temp_{job_id}.wav"
    audio_file.save(temp_filename)

    JOBS[job_id] = {
        "progress": 0,
        "status": "Pendiente",
        "completed": False,
        "txt_result": None,
        "json_result": None,
    }

    thread = threading.Thread(
        target=transcribe_in_background,
        args=(temp_filename, job_id, n_speakers, language_param),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id}), 200


@app.route("/transcribe/progress/<job_id>", methods=["GET"])
def transcribe_progress(job_id):
    job_info = JOBS.get(job_id)
    if not job_info:
        return jsonify({"error": "job_id desconocido"}), 404
    return jsonify(
        progress=job_info["progress"],
        status=job_info["status"],
        completed=job_info["completed"],
    ), 200


@app.route("/transcribe/result/<job_id>", methods=["GET"])
def transcribe_result(job_id):
    job_info = JOBS.get(job_id)
    if not job_info:
        return jsonify({"error": "job_id desconocido"}), 404
    return jsonify(
        txt_result=job_info["txt_result"],
        json_result=job_info["json_result"],
        completed=job_info["completed"],
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
