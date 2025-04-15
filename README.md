# Transcripción y agrupación de interlocutores en audio

## Descripción
Este proyecto ofrece:

- **Transcripción de audio** mediante OpenAI Whisper.
- **Agrupación por interlocutores** usando embeddings de Resemblyzer y clustering con KMeans.
- **Servicio web con Flask** que:
  - Recibe un archivo de audio y procesa la transcripción en un hilo en segundo plano.
  - Permite consultar el estado de progreso en tiempo real y obtener el resultado final vía endpoints de tipo polling.
  - Incluye un front-end (modo oscuro, responsive) para subir el archivo, mostrar el porcentaje de procesamiento y ver la transcripción final.
- Cuando no se especifica el número de interlocutores, se determina automáticamente usando el silhouette score.
- Si `n_speakers == 1`, se omiten los embeddings y clustering para ahorrar tiempo de procesamiento.

## Requisitos
- Python 3.7+
- Para GPU (opcional): se requieren drivers de NVIDIA y `torch` con CUDA compatible.
- Librerías principales:
  - `torch` (con soporte CUDA si dispones de GPU)
  - `openai-whisper`
  - `librosa`
  - `numpy`
  - `resemblyzer`
  - `scikit-learn`
  - `flask`
  - (Opcional) `python-dotenv` si usas `.env` para configuración

## Instalación básica
```bash
# Instala las librerías requeridas (sin versiones fijas):
pip install flask torch openai-whisper librosa numpy resemblyzer scikit-learn
