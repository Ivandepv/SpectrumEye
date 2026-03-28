import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

# 1. Iniciamos la comunicación con el módulo físico
try:
    sdr = RtlSdr()
except Exception as e:
    print(f"❌ Error al conectar con el SDR: {e}")
    print("Tip: Verifica que esté bien conectado o corre el script con 'sudo'.")
    exit()

# 2. Configuración de nuestro "Ojo" de radio
sdr.sample_rate = 2.048e6    # 2.048 Millones de muestras por segundo
sdr.center_freq = 1090e6   # Sintonizamos 433.92 MHz (IoT, Llaves, Sensores)
sdr.gain = 'auto'            # Dejamos que el chip ajuste el volumen

print("📡 Hardware conectado. Escuchando el aire durante una fracción de segundo...")

# 3. ¡LA CAPTURA REAL! 
# Pedimos 256,000 números complejos (aprox 1/8 de segundo de realidad)
muestras = sdr.read_samples(256 * 1024)

# 4. Apagamos el módulo para liberar el USB
sdr.close()

print(f"✅ ¡Éxito total! Tu Raspberry acaba de capturar {len(muestras)} muestras I/Q.")
print(f"🔍 Un vistazo a los datos puros (matemáticas físicas): \n{muestras[:4]}\n")

# 5. Visualización: Vamos a ver qué capturó
plt.figure(figsize=(10, 5))
# Usamos una función integrada para graficar el espectro de potencia
plt.psd(muestras, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6, color='cyan')

plt.title("Mi Primera Captura del Mundo Real - 433.92 MHz")
plt.xlabel("Frecuencia (MHz)")
plt.ylabel("Fuerza de la Señal (dB/Hz)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.style.use('dark_background') # Para que se vea profesional y hacker
plt.show()