import pathlib # Manejo de rutas de archivos
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
  # Contar número de palabras en texto (formato: texto, espacio)
  num_palabras = 0

  # 
  
  
  # Detener conteo al llegar a dos (optimiza al evitar contar todas las palabras)
  if num_palabras > 1:
    # 
    
  
  # Leer primera palabra (formato: texto, espacio)
  if num_palabras == 1:
    
  
  # Leer penúltima palabra (formato: espacio, texto, espacio)
  if num_palabras > 1:
    
  
  # Leer última palabra (formato: espacio, texto)
  if bool_terminado == True:
        

# Función para generar respuesta con un modelo Llama
def generar_respuesta(modelo_llama, texto_salida):
  # Llama al modelo con los parámetros especificados y retorna la respuesta
    return modelo_val(
        texto_entrada, # El texto de entrada (prompt) que se envía al modelo
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
  # Esperar voz del usuario

  
  # Capturar voz

  
  # Generar texto a partir de voz
  
  
  return texto_entrada

# Función para 
def ciclo_conversacion(modelo_whisper, modelo_llama, modelo_openvoice)
  # 1. Esperar hasta capturar voz para convertir a texto
  texto_entrada = capturar_voz(modelo_whisper)
  
  # 2. Introducir texto de voz como input para generar respuesta (procesar token por token)
  texto_salida = f"User: {texto_entrada}\nAssistant:" # Aplicar formato

  bool_terminado = False # Convertir a True al terminar de generar respuesta
  
  # Generar respuesta token por token
  for token_salida in generar_respuesta(modelo_llama, texto_salida):
        # Procesar cada token de la respuesta en streaming
        if 'choices' in token_salida:
            token_val = token_salida['choices'][0].get("text", "") # Extraer el texto del token

            # Evaluar si el texto ha terminado de generarse
            if :
              bool_terminado = True # Convertido a True al terminar de generar respuesta
    
        # 3. Convertir respuesta a voz (leer palabra a palabra)
        leer_respuesta(modelo_openvoice, texto_salida, bool_terminado)

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
            modelo_whisper = whisper.load_model(archivo_modelo)
          else:
              input('Place a .pt model in "model" folder and press Enter...')

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
              input('Place a .gguf model in "model" folder and press Enter...')

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
              input('Place a .pth model in "model" folder and press Enter...')
  
        # 4. Terminar while si se cargan todos los modelos
        if modelo_whisper and modelo_llama and modelo_openvoice:
            break

# PUNTO DE PARTIDA
cargar_modelos() # Cargar archivos de modelos al iniciar el programa

# Mensaje de iniciación
print("Start talking")

# Bucle principal del programa
while True:
  # 
  ciclo_conversacion(modelo_whisper, modelo_llama, modelo_openvoice)
