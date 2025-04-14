FROM python:3.9-slim

# 1. Instalar dependencias del sistema: ffmpeg, git, gcc, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    gcc \
    g++ \
    # Si quieres asegurarte de tener las herramientas de build completas,
    # añade build-essential:
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Crear y usar el directorio de la app
WORKDIR /app

# 3. Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto de la app
COPY . .

# 5. Exponer puerto (por si tu app Flask corre en 5000)
EXPOSE 5000

# 6. Comando para arrancar la app
CMD ["python", "app.py"]
