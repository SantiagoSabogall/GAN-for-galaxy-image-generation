# 📖 Manual Rápido (Para Principiantes)

Este documento te guiará paso a paso sobre cómo poner a funcionar el "Generador de Galaxias" sin que necesites saber programación avanzada ni entender cómo funcionan las intrincadas matemáticas de la inteligencia artificial.

## 1. ¿Qué necesitas tener instalado?
- **Python**: Necesitas tener Python en tu computadora (idealmente versión 3.9 o superior). Puedes descargarlo gratis desde [python.org](https://www.python.org/).
- A la hora de instalar Python, asegúrate de marcar siempre la casilla que dice **"Add Python to PATH"** (Añadir Python a las variables de entorno / PATH).

## 2. Preparando la carpeta
1. Abre tu **Terminal** (o "Símbolo del Sistema" o "PowerShell" en Windows).
2. Tienes que ir a la carpeta exacta donde está este proyecto. Puedes hacerlo escribiendo `cd` seguido de la ruta. Por ejemplo:
```bash
cd J:\Documentos\Dev\GAN-for-galaxy-image-generation
```

## 3. Instalando el motor del programa
Las piezas de inteligencia artificial como TensorFlow se llaman dependencias o librerías. Para que no tengas que instalar una por una, configuramos una lista automática. Solo escribe y presiona Enter:

```bash
pip install -r requirements.txt
```
*(Espera un ratito a que la barra de descarga de todo finalice).*

## 4. ¡A Entrenar a tu Inteligencia Artificial!
Ya está todo listo. Para que la IA comience a procesar el espacio y generar galaxias desde ceros, corre este comando:

```bash
python -m src.train
```

**¿Qué pasa después de presionar ENTER?**
- Tu computadora (en el fondo) comprobará si ya descargaste las fotos reales de galaxias de la base de datos "Galaxy Zoo" (si no lo has hecho, pesará unos GBs y lo descargará automáticamente).
- Luego verás que empieza a cargar texto con números como "*Epoch 1/10*". Cada época es un intento o ronda completa de la IA aprendiendo sobre la forma, bordes, luz y estructura de las galaxias reales.
- Si tu computadora hace algo de ruido (por los ventiladores de la tarjeta de video) o parece muy ocupada, **es completamente normal**, está calculando todo intensivamente. ¡Solo déjala trabajar solita!

## 5. Visualiza la Magia
No tienes que esperar a que terminen los números para ver cómo está progresando:
- Entra a la carpeta de tu proyecto, y verás una nueva subcarpeta llamada **`output`**. Entra allí y verás imágenes apareciendo gradualmente (se llaman `image_at_epoch_X.png`). Si abres la primera se verá como ruido o televisión sin señal, pero al abrir las siguientes verás la evolución de tu propio universo galáctico.
- En la carpeta **`checkpoints`**, la "memoria" e ideas de tu cerebro IA se autoguardan. Así, si cierras esto y regresas mañana, nada se habrá perdido.
- **AL FINALIZAR**: Cuando terminen todos los números y la terminal pare de cargarse, se generará un video corto con formato **GIF** (`galaxy_gan_evolution.gif`) de forma automática, para que puedas mandarlo por redes o mostrárselo a tus amigos y colegas como evidencia gráfica de que una computadora entrenada por ti aprendió a soñar en el espacio profundo.
