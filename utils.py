import cv2
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import numpy as np
import math 
from scipy.signal import butter, lfilter

def video_to_frames(video):
    '''
    Segmenta el video en frames
    
    Input: 
        video : path donde se encuentra el video que se quiere procesar
    
    Output:
        frames: lista con todos los frames del video
        nro_Frames: número de frames del video
        frame_rate: frames por segundo  
        vidHeight: número de filas de un frame
        vidWidth: número de columnas de un frame
    
    '''
    
    # Capturamos el video.
    vid = cv2.VideoCapture(video)
    frame_rate = vid.get(cv2.CAP_PROP_FPS)
    
    #Extraer info del video
    vidHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vidWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    nChannels = 3
    
    nro_Frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    
    if (vid.isOpened() == False):
        print('No fue cargado el video correctamente')
        
    while (vid.isOpened()):
        try:            
            
            for i in range(nro_Frames): # "endIndex - startIndex" -> Cantidad de piramides del stack.
                
                # Capturamos el frame.
                ret, frame = vid.read()

                if not ret:
                    # Release the Video Device if ret is false
                    vid.release()
                    # Message to be displayed after releasing the device
                    print("Released Video Resource")
                    break

                frames.append(frame)
                
        except KeyboardInterrupt:
            # Release the Video Device
            vid.release()
            # Message to be displayed after releasing the device
            print("Released Video Resource")

        # Release the Video Device
        vid.release()
    
    return frames, nro_Frames, frame_rate, vidHeight, vidWidth

def expand_gaussian_pyr(frame, levels):
    '''
    Arma la pirámide gaussiana de un frame del video con la cantidad de niveles especificados en levels
    Función auxiliar que se utiliza en "gaussian_video"
    
    Input: 
        frame: fotograma
        levels: nro de niveles de la piramide.
        
    Output:
        gaussian_pyramid_frame: piramide del fotograma correspondiente.
    
    '''
    temp = frame.copy() # frame.
    
    gaussian_pyramid_frame = [] # Piramide.
    
    #Cargamos el frame original
    gaussian_pyramid_frame.append(temp)
    
    for i in range(levels):
                
        rows, cols, _ = map(int, temp.shape)
        
        dst = cv2.pyrDown(temp) 
        
        gaussian_pyramid_frame.append(dst) # Agregamos la i-esima capa a la piramide.
                
        temp = dst.copy() # Ultimo frame.
        
    return gaussian_pyramid_frame

def gaussian_video(frames, levels):
    
    '''
    Arma la pirámide gaussiana de cada frame y luego toma el último nivel de cada pirámide
    para armar un video
    
    Input:
        frames: lista de frames
        
    Output:
        gen_gaussian_vid: video generado a partir del último nivel de cada pirámide
    '''
    
    for i in range(len(frames)):
        
        #Tomo un frame, armo su pirámide, me quedo con el último nivel
        frame = frames[i]
        pyr = expand_gaussian_pyr(frame, levels)
        last_lev = pyr[-1]        
        
        #Si es el primero, inicializo el video que voy a generar
        if i == 0:
            gen_gaussian_vid = np.zeros((len(frames), last_lev.shape[0], last_lev.shape[1], 3))
            
        gen_gaussian_vid[i] = last_lev
        
    return gen_gaussian_vid

def ideal_filter(frames_originales, gen_gaussian_vid, chromAttenuation, fl, fh, frame_rate, alpha):
    
    '''
    Toma el video gaussiano generado, le aplica un filtro pasabanda y amplifica.
    Luego lo suma a los frames originales. 
    Devuelve la suma del filtrado amplificado con el original
    
    Input:
        frames_originales: lista de frames originales del video (salida de video_to_frames)
        gen_gaussian_vid: video generado a partir del último nivel de cada pirámide (salida de gaussian_video)
        chromAttenuation: factor de atenuación para canales 2 y 3
        fl: frecuencia de corte baja
        fh: frecuencia de corte alta
        frame_rate: frames por segundo (salida de video_to_frames)
        alpha: factor de amplificación
    
    Output: 
        amplified_frames: suma del video filtrado amplificado con el video original
    '''
    
    #Real fft y definición de frecuencias
    fft_vid = fft.rfft(gen_gaussian_vid, axis = 0)
    freq = fft.rfftfreq(fft_vid.shape[0], d = 1.0 / frame_rate)
    
    # Máscara que mantiene los valores en la banda de interés
    mask = np.logical_and(freq > fl, freq < fh)  
    # Lleva a 0 todos los valores por afuera de la máscara
    fft_vid[~mask] = 0                
    
    # Inversa de real fft
    serie_filtrada = fft.irfft(fft_vid, axis = 0)  

    # Amplificación
    serie_amplificada = serie_filtrada * alpha
    
    # Aplicación de chromAttenuation
    serie_amplificada[:][:][:][1] *= chromAttenuation
    serie_amplificada[:][:][:][2] *= chromAttenuation

    # Resize para sumar a frames originales
    frames_filtrados = np.zeros((len(frames_originales), 
                                 frames_originales[0].shape[0], frames_originales[0].shape[1], 3)).astype('uint8')
    
    for i in range(len(frames_originales)):

        frames_filtrados[i] = cv2.resize(serie_amplificada[i], (frames_originales[0].shape[1], frames_originales[0].shape[0]))

    # Suma con los frames originales
    frames_filtrados += frames_originales
    
    #Todos los valores que quedaron por fuera los fuerzo dentro del rango
    frames_filtrados[frames_filtrados < 0] = 0
    frames_filtrados[frames_filtrados > 255] = 255
    
    return frames_filtrados

def frames_to_video(frames, nombre, frame_rate):
    
    '''
    Arma un video a partir de un conjunto de frames
    
    Input: 
        frames: lista de frames para armar el video
        nombre: nombre del video que se quiere poner
        frame_rate = frames por segundo
    
    Devuelve el video en carpeta.
    '''
    
    height, width,_ = frames[0].shape
    video = cv2.VideoWriter(str(nombre)+'.wmv',cv2.VideoWriter_fourcc(*'mp4v'),frame_rate,(width,height))

    for frame in frames:
        video.write(frame)
   
    video.release()


