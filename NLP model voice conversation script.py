import os # Módulo para acceder a funciones del sistema operativo
import pathlib # Manejo de rutas de archivos
import torch # Framework para manejar tensores y cargar modelos de deep learning
import sounddevice # Captura y reproducción de audio en tiempo real
import numpy # Procesamiento numérico eficiente, manejo de arrays para audio
import whisper # Conversión de voz a texto
import llama_cpp # Procesamiento de texto a respuesta
from openvoice import openvoice_cli # Conversión de respuesta en texto a voz

# VARIABLES
modelo_whisper = "" # Ruta hacia el archivo .pt
modelo_llama = "" # Ruta hacia el archivo .gguf
modelo_openvoice = "" # Ruta hacia el archivo .pth

# FUNCIONES
# Función para leer texto con OpenVoice
def leer_respuesta(modelo_openvoice, texto_salida, bool_terminado):
    # Variables estáticas para rastrear el estado entre llamadas
    if not hasattr(leer_respuesta, "ultimo_espacio"):
        leer_respuesta.ultimo_espacio = -1  # Posición del último espacio procesado
    
    # Buscar el último espacio en el texto actual
    ultimo_espacio_actual = texto_salida.rfind(" ")
    
    # 1. Primera palabra (formato "palabra ")
    if leer_respuesta.ultimo_espacio == -1 and ultimo_espacio_actual != -1:
        primera_palabra = texto_salida[:ultimo_espacio_actual].strip()
        
        if primera_palabra:
            openvoice_cli.speak(primera_palabra, modelo_openvoice)
          
        leer_respuesta.ultimo_espacio = ultimo_espacio_actual
    
    # 2. Palabra intermedia (formato " palabra ")
    elif ultimo_espacio_actual > leer_respuesta.ultimo_espacio:
        # Extraer la última palabra completa (entre espacios)
        inicio_palabra = leer_respuesta.ultimo_espacio + 1
        palabra_intermedia = texto_salida[inicio_palabra:ultimo_espacio_actual].strip()
      
        if palabra_intermedia:
            openvoice_cli.speak(palabra_intermedia, modelo_openvoice)
          
        leer_respuesta.ultimo_espacio = ultimo_espacio_actual
    
    # 3. Última palabra (formato " palabra" con bool_terminado=True)
    if bool_terminado and leer_respuesta.ultimo_espacio < len(texto_salida):
        ultima_palabra = texto_salida[leer_respuesta.ultimo_espacio + 1:].strip()
      
        if ultima_palabra:
            openvoice_cli.speak(ultima_palabra, modelo_openvoice)
          
        leer_respuesta.ultimo_espacio = len(texto_salida)  # Resetear para próxima interacción

# Función para generar respuesta con un modelo Llama
def generar_respuesta(modelo_llama, texto_prompt):
  # Llama al modelo con los parámetros especificados y retorna la respuesta
    return modelo_llama(
        texto_prompt, # El texto de entrada (prompt) que se envía al modelo
        max_tokens = 1024, # Máximo número de tokens que puede generar en la respuesta
        temperature = 0.6, # Controla la aleatoriedad (0 = determinista, 1 = muy aleatorio)
        top_p = 0.95, # Muestreo nucleus: considera tokens que sumen determinado porcentaje de probabilidad
        top_k = 40, # Considera solo los 40 tokens más probables en cada paso
        repeat_penalty = 1.1, # Penaliza la repetición de palabras (>1 reduce repetición)
        min_p = 0.05, # Probabilidad mínima que debe tener un token para ser considerado
        frequency_penalty = 0.1, # Penaliza tokens que aparecen frecuentemente
        presence_penalty = 0.1, # Penaliza tokens que ya aparecieron en el texto
        stream = True, # Retorna la respuesta token por token (streaming)
        stop = ["\n\n"], # Secuencias que detienen la generación
    )

# Función para capturar voz con Whisper
def capturar_voz(modelo_whisper):
    frecuencia_val = 16000 # frecuencia de muestreo
    umbral_db = 30 # nivel de energía en decibelios para iniciar grabación
    silencio_max = 3.0 # segundos de silencio para detener la grabación
    bloque_duracion = 0.1 # duración de cada bloque en segundos
    buffer_audio = [] # lista para almacenar los bloques de audio grabados
    silencio_val = 0 # contador de tiempo de silencio consecutivo
    grabando_val = False # estado de grabación, True si ya se detectó voz

    # Bucle principal para capturar audio hasta detectar silencio
    while True:
        bloque_val = sounddevice.rec(int(bloque_duracion * frecuencia_val), # tamaño del bloque en muestras
                                     samplerate = frecuencia_val, # frecuencia de muestreo
                                     channels = 1, # grabación en mono
                                     dtype = numpy.float32 # tipo de dato de las muestras
                                    )
        
        sounddevice.wait() # espera a que termine la grabación del bloque
        
        bloque_val = numpy.squeeze(bloque_val) # eliminar dimensiones extra
        
        # Convertir RMS a decibelios
        rms_val = numpy.sqrt(numpy.mean(bloque_val ** 2)) # energía del bloque
        
        decibelios_val = 20 * numpy.log10(rms_val + 1e - 9) # evitar log(0)

        # Detectar si se superó el umbral de voz
        if decibelios_val > umbral_db:
            grabando_val = True # iniciar grabación
            buffer_audio.extend(bloque_val) # guardar bloque en buffer
            
            silencio_val = 0 # resetear contador de silencio
        elif grabando_val:
            buffer_audio.extend(bloque_val) # seguir agregando bloques
            
            silencio_val += bloque_duracion # aumentar contador de silencio

            # Detener grabación si se supera el tiempo máximo de silencio
            if silencio_val >= silencio_max:
                
                break
    
    # Convertir el buffer a un array de numpy para Whisper
    audio_numpy = numpy.array(buffer_audio)

    # Transcribir audio a texto
    resultado_val = modelo_whisper.transcribe(audio_numpy)
    
    # Extraer el texto transcrito
    texto_entrada = resultado_val["text"]
    
    return texto_entrada

# Función para llevar a cabo conversación entre usuario y máquina
def ciclo_conversacion(modelo_whisper, modelo_llama, modelo_openvoice):
  # 1. Esperar hasta capturar voz para convertir a texto
  texto_entrada = capturar_voz(modelo_whisper)
  
  # 2. Introducir texto de voz como input para generar respuesta (procesar token por token)
  texto_prompt = f"User: {texto_entrada}\nAssistant:" # Aplicar formato

  bool_terminado = False # Convertir a True al terminar de generar respuesta
  
  # Generar respuesta token por token
  for token_salida in generar_respuesta(modelo_llama, texto_prompt):
        # Procesar cada token de la respuesta en streaming
        if 'choices' in token_salida:
            token_val = token_salida['choices'][0].get("text", "") # Extraer el texto del token

            # Evaluar si el texto ha terminado de generarse
            if token_salida['choices'][0]['finish_reason'] is not None:
              bool_terminado = True # Convertido a True al terminar de generar respuesta
    
        # 3. Convertir respuesta a voz (leer palabra a palabra)
        leer_respuesta(modelo_openvoice, token_val, bool_terminado)

# Función para 
def cargar_modelos():
    global modelo_whisper, modelo_llama, modelo_openvoice
  
    carpeta_modelos = pathlib.Path("models")
    
    # Crear carpeta si no existe
    if not carpeta_modelos.exists():
        carpeta_modelos.mkdir(exist_ok = True)
    
        input('Place model files in "models" folder and press Enter...')
    
    # Ruta del archivo
    archivo_modelo = None
    
    # Buscar y cargar modelos en un bucle infinito hasta encontrarlos
    while True:
        # 1. Buscar y cargar modelo Whisper
        if not modelo_whisper:
          # Buscar archivos .pt en la carpeta
          for archivo_iter in carpeta_modelos.glob("*.pt"):
              if archivo_iter.is_file():
    
                  archivo_modelo = archivo_iter # Asignar el primer archivo encontrado
    
                  break
    
          # Si se encontró un modelo, cargarlo
          if archivo_modelo:
            # Cargar modelo Whsiper
            modelo_whisper = torch.load(archivo_modelo)
          else:
              input('Place a .pt model in "models" folder and press Enter...')
    
        # 2. Buscar y cargar modelo LLaMA
        if not modelo_llama:
          # Buscar archivos .gguf en la carpeta
          for archivo_iter in carpeta_modelos.glob("*.gguf"):
              if archivo_iter.is_file():
    
                  archivo_modelo = archivo_iter # Asignar el primer archivo encontrado
    
                  break
    
          # Si se encontró un modelo, cargarlo
          if archivo_modelo:
            # Cargar modelo LLaMA
            modelo_llama = llama_cpp.Llama(
                          model_path = str(archivo_modelo), # Ruta del archivo del modelo
                          n_ctx = 32768, # Tamaño del contexto (memoria del modelo)
                          n_gpu_layers = -1, # Usar GPU para todas las capas (-1 = todas)
                          n_threads = os.cpu_count(), # Usar todos los núcleos de la CPU
                          n_batch = 512, # Tamaño del lote para procesamiento
                          use_mmap = True, # Usar memory mapping para cargar el modelo
                          use_mlock = True, # Bloquear memoria para evitar swapping
                          verbose = False, # No mostrar información detallada durante la carga
                          f16_kv = True, # Usar precisión float16 para cache de key-value
                          logits_all = False, # No calcular logits para todos los tokens
                          vocab_only = False, # No cargar solo el vocabulario
                          rope_scaling_type = llama_cpp.LLAMA_ROPE_SCALING_TYPE_LINEAR, # Tipo de escalado RoPE
                      )
          else:
              input('Place a .gguf model in "models" folder and press Enter...')
    
        # 3. Buscar y cargar modelo OpenVoice
        if not modelo_openvoice:
          # Buscar archivos .pt en la carpeta
          for archivo_iter in carpeta_modelos.glob("*.pth"):
              if archivo_iter.is_file():
    
                  archivo_modelo = archivo_iter # Asignar el primer archivo encontrado
    
                  break
    
          # Si se encontró un modelo, cargarlo
          if archivo_modelo:
            # Cargar modelo OpenVoice
            modelo_openvoice = torch.load(archivo_modelo, map_location = "cpu")
          else:
              input('Place a .pth model in "models" folder and press Enter...')
    
        # 4. Terminar while si se cargan todos los modelos
        if modelo_whisper and modelo_llama and modelo_openvoice:
            break

# PUNTO DE PARTIDA
cargar_modelos() # Cargar archivos de modelos al iniciar el programa

# Mensaje de iniciación
print("Start talking")

# Bucle principal del programa
while True:
  # Iniciar el ciclo de captura de voz y lectura de respuesta generada
  ciclo_conversacion(modelo_whisper, modelo_llama, modelo_openvoice)
