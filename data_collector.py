import numpy as np 
import matplotlib.pyplot as plt
import os
from rtlsdr import RtlSdr
import random

def generate_t_sampling(fs, duration): #we need the frequency
    period = 1/fs #signal period/step/numerical distance between samples
    return np.arange(0, duration, period)

def capture_real_signal(sdr, num_samples):
    samples = sdr.read_samples(num_samples)
    return samples

def sample_library():
    # 1. Nuestra base de datos de frecuencias (en Hz)
    # Formato: "nombre_categoria" : (frecuencia_minima, frecuencia_maxima)
    frequency_db = {
        "radio_fm": (88e6, 108e6),          # FM Comercial (WBFM)
        "air_traffic": (108e6, 137e6),      # Aerial control tower 
        "noaa" : (137e6, 138e6), #Meteorological Satellites
        "local_repeaters" : (144e6, 148e6), #Amateur Radio (2m Band): People talking over long distances, local repeaters.
        "maritime": (156e6, 174e6),         # Maritime traffic and emergencies (VHF)
        "short_range_devices" : (315e6, 433.05e6), # Car keys, electric gates, wireless doorbells, baby monitors, temperature sensors.
        "wireless_controllers": (433.05e6, 434.79e6),   # MORE SPECIFIC short-range devices
        "walkie_talkie": (446e6, 462e6),    # Walkie-Talkies (UHF)
        "cellular_network" : (850e6, 960e6), #Cellular Networks (GSM/LTE/5G) and LoRaWAN: Cell towers sending data and long-range industrial sensors.
        "aircraft_tracking": (1090e6, 1090e6)     # Comercial Aircraft
    }
    
    # 2. Le mostramos las opciones al usuario
    print("\nFREQUENCIES AVAILABLES:")
    for key in frequency_db.keys():
        print(f" - {key}")
        
    user_selection = input("\nWhat do you wish to hear?: ").strip().lower()

    if user_selection in frequency_db:
        min_freq, max_freq = frequency_db[user_selection]
        
        target_freq = random.uniform(min_freq, max_freq)
        
        print(f"Tuning in {user_selection} signals in {target_freq / 1e6:.2f} MHz...")
        return target_freq, user_selection
    else:
        print("I couldn't understand what you wrote. Please try again")
        return None, None
    
## Primero en el dominio del tiempo ##
def time_domain_visualization(t, original_signal, noisy_signal):
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.plot(t, noisy_signal, label='WIRELESS KEY SIGNALS') # CORREGIDO
    plt.plot(t, original_signal, label='Original signal', color='red', linewidth=2) # CORREGIDO
    plt.legend()
    plt.title('TIME DOMAIN WIRELESS KEY SIGNAL (with noise)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
def fd_color(real_world_signal,fs,filename):
        ### Spectrogram visualization in color###
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(real_world_signal, NFFT=256, Fs=fs, cmap='jet') #NFFT (Number of points for FFT o comúnmente referida al análisis FFT con ventanas
    
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    plt.savefig(filename, dpi=100)
    
    plt.close()
    
def fd_gray(real_world_signal,fs,filename):
        ### Spectrogram visualization in gray###
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(real_world_signal, NFFT=256, Fs=fs, cmap='gray') #NFFT (Number of points for FFT o comúnmente referida al análisis FFT con ventanas
    
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    plt.savefig(filename, dpi=100)
    
    plt.close()

def create_dataset(n_image, num_samples, count_real_data, dataset_name, color_dataset):
    if color_dataset == "grayscale":
        base_dir = gray_path_dir
    else:
        base_dir = color_path_dir
    path_directory = os.path.join(base_dir, dataset_name)
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(path_directory, exist_ok=True)
    
    existing_files = [f for f in os.listdir(path_directory) if f.endswith('.png')]
    
    count_real_data = len(existing_files)
    
    if color_dataset == "grayscale":
        try:
            for n in range(n_image):
                real_world_signal = capture_real_signal(sdr, num_samples)
                
                # B. Nombramos el archivo
                filename = f"sample_{count_real_data:03d}.png"
                path = os.path.join(path_directory, filename)
                
                fd_gray(real_world_signal, fs, path)
                print(f"Saving: {filename}")
                
                count_real_data += 1

        except Exception as e:
            print(f"The capture was not possible: {e}")

        finally:
            sdr.close()
            print("Sampling finish. Dataset generated.")
    else:
        try:
            for n in range(n_image):
                real_world_signal = capture_real_signal(sdr, num_samples)
                
                # B. Nombramos el archivo
                filename = f"sample_{count_real_data:03d}.png"
                path = os.path.join(path_directory, filename)
                
                fd_color(real_world_signal, fs, path)
                print(f"Saving: {filename}")
                
                count_real_data += 1

        except Exception as e:
            print(f"The capture was not possible: {e}")

        finally:
            sdr.close()
            print("Sampling finish. Dataset generated.")

### File organizer ###
base_dir = "dataset_rf" #The main directory
gray_path_dir = os.path.join(base_dir, "dataset_gray") #Here you create each directory that you need
color_path_dir =  os.path.join(base_dir, "dataset_color")


os.makedirs(base_dir, exist_ok=True)
os.makedirs(gray_path_dir, exist_ok=True)
os.makedirs(color_path_dir, exist_ok=True)


sdr = RtlSdr()
sdr.sample_rate = 2.048e6    # 2.048 MHz de ancho de banda
sdr.center_freq, dataset_name = sample_library()  
sdr.gain = 'auto'            # Ganancia automáti

fs = sdr.sample_rate
n_image = 500 #Change the number of image to whatever quantity you need
num_samples = 256 * 1024

color_dataset = input("Do you want to generate the sampling in color or grayscale? ").lower()
count_real_data = 0

create_dataset(n_image,  num_samples, count_real_data, dataset_name,color_dataset)


