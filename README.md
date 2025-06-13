# Transcriptor y Agrupador de Interlocutores en Audio

**Resumen:**  
Esta aplicación web permite transcribir archivos de audio y agrupar las intervenciones por hablante de forma automática. Utiliza OpenAI Whisper para la transcripción y Resemblyzer + KMeans para identificar y separar los distintos interlocutores. El procesamiento se realiza en segundo plano y puedes consultar el progreso y el resultado desde la propia web. Todo funciona fácilmente desde un contenedor Docker, sin necesidad de instalar nada en tu sistema.

---

## Instalación y uso rápido (con Docker)

1. **Construye la imagen Docker:**
   ```bash
   docker build -t audio-to-text .
   ```

2. **Ejecuta el contenedor:**
   ```bash
   docker run -p 5000:5000 audio-to-text
   ```

3. **Abre la aplicación en tu navegador:**  
   [http://localhost:5000](http://localhost:5000)

---

## (Opcional) Instalación manual para desarrolladores

Si prefieres ejecutar el proyecto fuera de Docker (por ejemplo, para desarrollo o depuración):

1. **Clona el repositorio y entra en la carpeta:**
   ```bash
   git clone <URL_DEL_REPO>
   cd audio_to_text
   ```

2. **Instala ffmpeg (si no lo tienes):**
   ```bash
   sudo apt-get update && sudo apt-get install ffmpeg
   ```

3. **Crea un entorno virtual (opcional):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Instala las dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Ejecuta la aplicación:**
   ```bash
   python app.py
   ```
   Accede a [http://localhost:5000](http://localhost:5000) en tu navegador.

---

## Características

- Transcripción automática de audio con Whisper.
- Agrupación de intervenciones por hablante usando Resemblyzer y KMeans.
- Detección automática del número de hablantes (o puedes indicarlo manualmente).
- Interfaz web moderna (modo oscuro, responsive).
- Procesamiento en segundo plano y endpoints REST para integración.
- Soporte para GPU (CUDA) si está disponible.
- Dockerfile listo para despliegue.

## Endpoints principales

- `GET /`  
  Interfaz web.

- `POST /transcribe`  
  Sube un archivo de audio. Devuelve un `job_id`.

- `GET /transcribe/progress/<job_id>`  
  Consulta el progreso del procesamiento.

- `GET /transcribe/result/<job_id>`  
  Obtiene el resultado final (texto y JSON).

## Créditos

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- [Flask](https://flask.palletsprojects.com/)
- [Bootstrap](https://getbootstrap.com/)

---

- **Soporte para más idiomas** y selección manual del idioma de transcripción.
- **Descarga directa** de la transcripción en formatos TXT, DOCX o PDF.
- **Visualización de segmentos de audio** con marcas de tiempo y saltos rápidos.
- **Identificación automática de género o nombre de hablantes** (si es posible).
- **Historial de transcripciones** y gestión de trabajos anteriores.
- **Autenticación de usuarios** para uso multiusuario.
- **Integración con servicios en la nube** (Google Drive, Dropbox, etc.).
- **Notificaciones por email** al finalizar la transcripción.
- **Mejoras en la interfaz**: edición manual de segmentos, corrección de errores, etc.
- **Despliegue fácil en servicios cloud** (Heroku, AWS,

**¡Contribuciones y sugerencias son bienvenidas!**