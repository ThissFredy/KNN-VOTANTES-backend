# 1. Imagen base de Python
# Usamos 'slim' para que sea una imagen ligera
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias
# Copiamos primero el 'requirements.txt' para aprovechar el caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el proyecto
# Copia las carpetas 'app/' y 'data/' al directorio /app
COPY . .

# 5. Exponer el puerto
# Informa a Docker que el contenedor escuchará en el puerto 8000
EXPOSE 8000

# 6. Comando de ejecución
# El comando para iniciar la API cuando el contenedor arranque
# '--host 0.0.0.0' es crucial para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]