import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

#############
# Generales #
#############

# Extrae y normaliza el canal de luminancia a [0, 1].
def extraer_luminancia(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3 and img.shape[2] in [3, 4]:
        # Conversión Estándar ITU-R BT.601 / BT.709
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        luminance = img.copy()
    # Normalización a [0, 1]    
    min_val, max_val = luminance.min(), luminance.max()
    if max_val > min_val:
        luminance = (luminance - min_val) / (max_val - min_val)
    else:
        luminance = np.zeros_like(luminance)
        
    return luminance

# Retorna el kernel B3-Spline unidimensional normalizado
def kernel_b3():
    return np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0

# suavizar y reducir tamaño por factor 2
def downsample2d(image):
    # suavizar con filtro B3-spline
    h = kernel_b3()
    suavizado = ndi.convolve1d(image, h, axis=0, mode='reflect')
    suavizado = ndi.convolve1d(suavizado, h, axis=1, mode='reflect')
    # tomar cada segundo pixel en cada eje
    return suavizado[::2, ::2]

# Intercala ceros y suaviza con filtro B3-spline para aumentar tamaño
def upsample2d(image, target_shape):
    h = kernel_b3()
    H, W = target_shape
    upsampled = np.zeros((2 * image.shape[0], 2 * image.shape[1]), dtype=image.dtype)
    upsampled[::2, ::2] = image
    upsampled = upsampled[:H, :W]
    
    # Factor 2 en cada eje para preservar escala
    h = 2.0 * h
    expandida = ndi.convolve1d(upsampled, h, axis=0, mode='reflect')
    expandida = ndi.convolve1d(expandida, h, axis=1, mode='reflect')
    return expandida

#######################
# Pirámide Laplaciana #
#######################

# Construye la Pirámide Laplaciana entregando capas + residual
def piramide_laplaciana(image, niveles=5):
    """
    Construye una Pirámide Laplaciana de 'niveles' capas de detalle más residual.
    Retorna:
        details: lista de 5 capas de detalle [d1, d2, d3, d4, d5]
        residual: c5 (aproximación gruesa final)
    """
    # Construcción de la pirámide gaussiana
    piramide_gauss = [image.copy()]
    curr = image.copy()
    for _ in range(niveles):
        curr = downsample2d(curr)
        piramide_gauss.append(curr)

    # Construcción de la pirámide laplaciana
    piramide_laplaciana = []
    for nivel in range(niveles):
        expandida = upsample2d(piramide_gauss[nivel+1], piramide_gauss[nivel].shape)
        diferencia = piramide_gauss[nivel] - expandida
        piramide_laplaciana.append(diferencia)

    # capa residual final (imagen suavizada más gruesa)
    residual = piramide_gauss[-1]
    return piramide_laplaciana, residual

# Reconstruir imagen a partir de la pirámide laplaciana
def reconstruccion_laplaciana(detalles, residual):
    # partimos de la capa residual y vamos sumando las capas de detalle
    # de menos a más fina (de atrás hacia adelante)
    curr = residual.copy()
    for capa in reversed(range(len(detalles))):
        expandida = upsample2d(curr, detalles[capa].shape)
        curr = detalles[capa] + expandida
    return curr

########################
# Transformada Starlet #
########################


def transformada_starlet(image, niveles=5):
    b3 = kernel_b3()
    curr = image.copy()
    capas = []
    for j in range(niveles):
        # kernel con ceros entre los coeficientes para el nivel j
        step = 2**j
        kernel_len = 4 * step + 1
        h_j = np.zeros(kernel_len, dtype=np.float64)
        h_j[::step] = b3
        
        sig = ndi.convolve1d(curr, h_j, axis=0, mode='reflect')
        sig = ndi.convolve1d(sig, h_j, axis=1, mode='reflect')
        
        detalle = curr - sig
        capas.append(detalle)
        curr = sig
        
    residual = curr
    return capas, residual

# Reconstruir imagen a partir de la transformada Starlet
def reconstruct_starlet_transform(detalles, residual):
    reconstruccion = residual.copy()
    for d in detalles:
        # sumamos cada capa de detalle a la reconstrucción
        reconstruccion = reconstruccion + d
    return reconstruccion


######################################################
# Métricas y funciones de experimentación y análisis #
######################################################

# Error cuadrático medio entre dos imágenes
def calculate_rmse(img1, img2):
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))

# Aplicar umbral duro relativo a cada capa de detalle y calc % coef conservados
def umbral_duro(detalles, alpha):
    detalles_umbral = []
    porcentajes_conservados = []
    for d in detalles:
        max_dj = np.max(np.abs(d))
        Tj = alpha * max_dj
        mask = np.abs(d) >= Tj
        d_umbral = np.where(mask, d, 0.0)
        
        porcentaje = (np.count_nonzero(mask) / d.size) * 100.0
        detalles_umbral.append(d_umbral)
        porcentajes_conservados.append(porcentaje)
        
    return detalles_umbral, porcentajes_conservados

# Aplicar factor multiplicativo a cada capa de detalle
def aplicar_factor_mult(detalles, factores):
    capas_mod = []
    for d, f in zip(detalles, factores):
        capas_mod.append(d * f)
    return capas_mod


# Funcion auxiliar para graficar las capas de detalle y el residual
def plot_capas(detalles, residual, title_prefix=""):
    num_layers = len(detalles)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    
    for j in range(num_layers):
        d = detalles[j]
        lim = np.percentile(np.abs(d), 99.5) if np.max(np.abs(d)) > 0 else 1.0
        im = axes[j].imshow(d, cmap='gray', vmin=-lim, vmax=lim)
        axes[j].set_title(f"{title_prefix} Capa {j+1}\nDim: {d.shape[0]}x{d.shape[1]}")
        axes[j].axis('off')
        plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        
    im_res = axes[num_layers].imshow(residual, cmap='gray', vmin=0, vmax=1)
    axes[num_layers].set_title(f"{title_prefix} Residual\nDim: {residual.shape[0]}x{residual.shape[1]}")
    axes[num_layers].axis('off')
    plt.colorbar(im_res, ax=axes[num_layers], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig, axes