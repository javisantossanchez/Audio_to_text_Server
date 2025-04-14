import os
import uuid
import time
import threading

import torch
import whisper
import librosa
import numpy as np

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import timedelta

###################################################
# Cargar variables de entorno desde .env
###################################################
load_dotenv()

NAS_DIRECTORY = os.getenv("NAS_DIRECTORY", "nas_audio")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# Detectar GPU vs CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ GPU detectada. Usando CUDA.")
else:
    DEVICE = "cpu"
    print("⚠️ No se detecta GPU. Usando CPU.")

# Iniciar la app Flask
app = Flask(__name__)

# Cargar (lazy) el modelo Whisper
print(f"🔊 Cargando modelo Whisper [{WHISPER_MODEL_NAME}] en {DEVICE}...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
print("✅ Modelo cargado.")


def transcribe_audio(audio_path: str, n_speakers: int = None):
    """
    Transcribe un audio usando Whisper + clustering por locutor.
    Devuelve un dict con 'txt_result' y 'json_result'.
    """
    # 1. Transcribir con Whisper
    result = whisper_model.transcribe(audio_path, verbose=False)
    segments = result["segments"]

    # 2. Generar embeddings con Resemblyzer
    wav, sr = librosa.load(audio_path, sr=16000)
    encoder = VoiceEncoder()
    
    embeddings = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        audio_slice = wav[start_sample:end_sample]

        if len(audio_slice) < sr * 0.5:
            # Segmento demasiado corto, asignar vector neutro
            embeddings.append(np.zeros(256))
        else:
            embedding = encoder.embed_utterance(audio_slice)
            embeddings.append(embedding)

    # 3. Determinar número óptimo de clusters (solo si n_speakers no viene definido)
    if n_speakers is None:
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

    # 4. Clustering final
    X = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=n_speakers, random_state=0).fit(X)
    labels = kmeans.labels_

    # 5. Agrupar intervenciones consecutivas
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

    # Añadir el último
    if texto_actual:
        aggregated_segments.append((inicio_actual, fin_actual, ultimo_speaker, texto_actual.strip()))

    # 6. Construir el resultado en texto
    lines = []
    for start, end, speaker, texto in aggregated_segments:
        start_time = str(timedelta(seconds=int(start)))
        end_time = str(timedelta(seconds=int(end)))
        line = f"[{start_time} - {end_time}] {speaker}: {texto}"
        lines.append(line)
    txt_result = "\n".join(lines)

    # 7. Construir resultado en JSON
    json_result = {
        "speakers_detected": n_speakers,
        "segments": []
    }
    for start, end, speaker, texto in aggregated_segments:
        json_result["segments"].append({
            "start": float(start),
            "end": float(end),
            "speaker": speaker,
            "text": texto
        })

    return {
        "txt_result": txt_result,
        "json_result": json_result
    }


@app.route("/")
def index():
    """
    Renderiza la página inicial con un formulario
    para subir el archivo y ver la transcripción.
    """
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    """
    Recibe el archivo vía POST y retorna la transcripción
    en JSON (incluye también el texto agrupado).
    """
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return jsonify({"error": "No se recibió ningún archivo 'audio_file'"}), 400

    # Leer n_speakers si se pasa desde el frontend
    n_speakers_str = request.form.get("n_speakers", None)
    n_speakers = int(n_speakers_str) if (n_speakers_str and n_speakers_str.isdigit()) else None

    # Guardar temporalmente
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    audio_file.save(temp_filename)

    try:
        # Transcribir
        result = transcribe_audio(temp_filename, n_speakers=n_speakers)
        return jsonify({
            "transcription_txt": result["txt_result"],
            "transcription_json": result["json_result"]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Limpieza
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


###############################
# (OPCIONAL) MONITOREAR NAS
###############################
def monitor_nas_directory():
    print(f"📂 Iniciando monitor de NAS en {NAS_DIRECTORY}...")
    processed = set()

    while True:
        try:
            # Recorrer archivos en el NAS
            for filename in os.listdir(NAS_DIRECTORY):
                if not filename.lower().endswith((".wav", ".mp3", ".m4a")):
                    continue

                filepath = os.path.join(NAS_DIRECTORY, filename)
                if filepath not in processed:
                    processed.add(filepath)
                    print(f"🔍 Nuevo archivo detectado: {filepath}")
                    # Transcribir y guardar
                    try:
                        r = transcribe_audio(filepath)
                        # Guardar .txt
                        txt_path = filepath + ".txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(r["txt_result"])
                        # Guardar .json
                        import json
                        json_path = filepath + ".json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(r["json_result"], f, ensure_ascii=False, indent=2)
                        print(f"✅ Transcripción generada: {txt_path} y {json_path}")
                    except Exception as e:
                        print(f"❌ Error transcribiendo {filepath}: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Error en monitor_nas_directory: {e}")
            time.sleep(5)

# Iniciar el monitor en segundo plano si existe la carpeta
if os.path.isdir(NAS_DIRECTORY):
    monitor_thread = threading.Thread(target=monitor_nas_directory, daemon=True)
    monitor_thread.start()
else:
    print(f"⚠️ Directorio NAS no encontrado: {NAS_DIRECTORY}. Monitor no iniciado.")


if __name__ == "__main__":
    # Puedes leer FLASK_RUN_PORT del .env o usar un valor por defecto
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
