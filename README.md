# Transcripción y agrupación de interlocutores en audio

## Descripción
Este programa transcribe un archivo de audio, agrupa las intervenciones por interlocutores y guarda la transcripción en un archivo de texto. Utiliza Whisper para la transcripción, Resemblyzer para generar embeddings y KMeans para la clusterización. Si no se especifica el número de interlocutores, se determina automáticamente usando el silhouette score.

## Requisitos
- Python 3.7+
- Bibliotecas: `argparse`, `whisper`, `librosa`, `numpy`, `resemblyzer`, `scikit-learn`, `datetime`

## Instalación
```bash
pip install openai-whisper librosa numpy resemblyzer scikit-learn
