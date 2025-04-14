import os
import uuid
import threading
from flask import Flask, request, jsonify, render_template
import torch
import whisper
import librosa
import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import timedelta
import subprocess


app = Flask(__name__)

###############################################
# DETECCIÓN DE GPU vs CPU
###############################################
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ GPU detectada. Usando CUDA.")
else:
    DEVICE = "cpu"
    print("⚠️ No se detecta GPU. Usando CPU.")

###############################################
# CARGAR MODELO WHISPER
###############################################
print("Cargando modelo Whisper ('base') en", DEVICE, "...")
whisper_model = whisper.load_model("base", device=DEVICE)
print("✅ Modelo Whisper listo.")

def extract_audio(input_file: str, output_file: str, sr: int = 16000):
    """
    Extrae la pista de audio de 'input_file' y la convierte a WAV mono,
    sample rate 'sr'. Requiere ffmpeg instalado.
    """
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vn",                # sin video
        "-acodec", "pcm_s16le",  # salida en raw PCM 16 bits
        "-ar", str(sr),      # sample rate
        "-ac", "1",          # mono
        output_file,
        "-y"                 # overwrite
    ]
    subprocess.run(cmd, check=True)


###############################################
# MEMORIA DE TRABAJOS (job_id -> estado)
###############################################
JOBS = {}  # { job_id: {"progress": int, "status": str, "completed": bool, "txt_result": str, "json_result": dict}}

###############################################
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
###############################################
def transcribe_in_background(audio_path: str, job_id: str, n_speakers: int = None):
    """
    Realiza la transcripción real con Whisper + Resemblyzer + KMeans,
    actualizando el estado (progreso) en JOBS[job_id].

    Si n_speakers == 1, salta la parte de embeddings y clustering,
    asumiendo que todos los segmentos pertenecen a un único locutor.
    """
    job_info = JOBS[job_id]
    try:
        # 1) Iniciando...
        _update_job(job_id, 5, "Audio cargado, iniciando transcripción...\n")
        _update_job(job_id, 20, "Transcribiendo...\n")

        # Transcripción con Whisper
        result = whisper_model.transcribe(audio_path, verbose=False)
        segments = result["segments"]

        # Si n_speakers == 1, evitamos embeddings y clustering
        if n_speakers == 1:
            # Saltamos directamente a agrupar todos los segmentos con la misma etiqueta
            _update_job(job_id, 50, "Un solo hablante. Saltando embeddings/clustering...\n")

            # Etiquetamos todos los segmentos con "SPEAKER_0"
            labels = [0] * len(segments)

        else:
            # Aquí seguimos el flujo habitual con embeddings y clustering
            _update_job(job_id, 75, "Generando embeddings con Resemblyzer...\n")

            wav, sr = librosa.load(audio_path, sr=16000)
            encoder = VoiceEncoder()
            embeddings = []
            for seg in segments:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                audio_slice = wav[start_sample:end_sample]

                if len(audio_slice) < sr * 0.5:
                    embeddings.append(np.zeros(256))
                else:
                    embedding = encoder.embed_utterance(audio_slice)
                    embeddings.append(embedding)

            # Determinamos número de clusters solo si no viene definido
            if n_speakers is None:
                _update_job(job_id, 85, "Determinando número de locutores...")
                X = np.vstack(embeddings)
                best_k = 2
                best_score = -1
                for k in range(2, 10):
                    kmeans_test = KMeans(n_clusters=k, random_state=0).fit(X)
                    labels_test = kmeans_test.labels_
                    score = silhouette_score(X, labels_test) if k > 1 else -1
                    if score > best_score:
                        best_score = score
                        best_k = k
                n_speakers = best_k
                _update_job(job_id, 85, f"Mejor k: {n_speakers} locutores...")
            else:
                _update_job(job_id, 85, f"Clustering para {n_speakers} locutores...")

            # Clustering con KMeans
            _update_job(job_id, 90, f"Realizando clustering...")
            X = np.vstack(embeddings)
            kmeans = KMeans(n_clusters=n_speakers, random_state=0).fit(X)
            labels = kmeans.labels_

        # 2) Agrupar intervenciones
        _update_job(job_id, 95, "Agrupando intervenciones...")
        aggregated_segments = []
        ultimo_speaker = None
        texto_actual = ""
        inicio_actual = None
        fin_actual = None

        for i, seg in enumerate(segments):
            speaker = f"SPEAKER_{labels[i]}"
            text = seg["text"].strip()
            start = seg["start"]
            end = seg["end"]

            if speaker == ultimo_speaker:
                texto_actual += " " + text
                fin_actual = end
            else:
                if ultimo_speaker is not None:
                    aggregated_segments.append((inicio_actual, fin_actual, ultimo_speaker, texto_actual.strip()))
                ultimo_speaker = speaker
                texto_actual = text
                inicio_actual = start
                fin_actual = end

        if texto_actual:
            aggregated_segments.append((inicio_actual, fin_actual, ultimo_speaker, texto_actual.strip()))

        # 3) Construir resultado final
        txt_result = []
        for start, end, speaker, texto in aggregated_segments:
            start_time = str(timedelta(seconds=int(start)))
            end_time = str(timedelta(seconds=int(end)))
            line = f"[{start_time} - {end_time}] {speaker}: {texto}"
            txt_result.append(line)
        txt_result = "\n".join(txt_result)

        json_result = {
            "speakers_detected": n_speakers if n_speakers else 1,  # n_speakers final
            "segments": []
        }
        for start, end, speaker, texto in aggregated_segments:
            json_result["segments"].append({
                "start": float(start),
                "end": float(end),
                "speaker": speaker,
                "text": texto
            })

        # Finalizar
        job_info["progress"] = 100
        job_info["status"] = "Completado"
        job_info["completed"] = True
        job_info["txt_result"] = txt_result
        job_info["json_result"] = json_result

    except Exception as e:
        job_info["progress"] = 100
        job_info["status"] = f"Error: {e}"
        job_info["completed"] = True
        job_info["txt_result"] = f"ERROR: {e}"
        job_info["json_result"] = {"error": str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def _update_job(job_id: str, progress: int, status: str):
    """
    Actualiza el porcentaje y estado de un job
    """
    job_info = JOBS[job_id]
    job_info["progress"] = progress
    job_info["status"] = status


###############################################
# ENDPOINTS FLASK
###############################################
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return jsonify({"error": "No se envió ningún archivo 'audio_file'"}), 400

    n_speakers_str = request.form.get("n_speakers")
    n_speakers = None
    if n_speakers_str and n_speakers_str.isdigit():
        n_speakers = int(n_speakers_str)

    job_id = str(uuid.uuid4())
    temp_filename = f"temp_{job_id}.wav"
    audio_file.save(temp_filename)

    # Crear registro del job
    JOBS[job_id] = {
        "progress": 0,
        "status": "Pendiente",
        "completed": False,
        "txt_result": None,
        "json_result": None
    }

    # Lanzar hilo en background
    thread = threading.Thread(target=transcribe_in_background, args=(temp_filename, job_id, n_speakers))
    thread.start()

    return jsonify({"job_id": job_id}), 200


@app.route("/transcribe/progress/<job_id>", methods=["GET"])
def transcribe_progress(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "job_id desconocido"}), 404
    job_info = JOBS[job_id]
    return jsonify({
        "progress": job_info["progress"],
        "status": job_info["status"],
        "completed": job_info["completed"]
    }), 200


@app.route("/transcribe/result/<job_id>", methods=["GET"])
def transcribe_result(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "job_id desconocido"}), 404
    job_info = JOBS[job_id]
    return jsonify({
        "txt_result": job_info["txt_result"],
        "json_result": job_info["json_result"],
        "completed": job_info["completed"]
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
