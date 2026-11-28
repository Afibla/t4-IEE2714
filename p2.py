import numpy as np
import cv2
import zipfile
from io import BytesIO

# Cargar una imagen HDR desdeun zip
def cargar_img(source_path_or_name, zip_path='archivos_t4.zip'):
    # Intentar cargar desde zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            target_name = next(
                (n for n in zf.namelist() if n.lower().endswith(source_path_or_name.lower())),
                None
            )
            if target_name is not None:
                data = zf.read(target_name)
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
    except Exception:
        pass
        
    raise FileNotFoundError(f"No se pudo cargar la imagen HDR '{source_path_or_name}' desde disco ni desde '{zip_path}'.")

# igual que en capsula 
def luminancia(E):
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 3 and E.shape[2] in [3, 4]:
        return 0.177 * E[..., 0] + 0.813 * E[..., 1] + 0.011 * E[..., 2]
    return E.copy()

def reescalar(g, p_bajo=0.5, p_alto=99.5):
    # Lleva los percentiles p_bajo y p_alto a 0 y 1, recortando los extremos
    lo, hi = np.percentile(g, [p_bajo, p_alto])
    if hi > lo:
        return np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(g)

def H_homomorfico(forma, gamma_H=1.5, gamma_L=0.25, D0=40, c=1.0):
    # Genera la respuesta en frecuencia H(u, v) del filtro homomórfico.
    P, Q = forma
    v, u = np.mgrid[0:P, 0:Q]
    D2 = (u - Q / 2) ** 2 + (v - P / 2) ** 2          # D(u, v)^2, distancia al centro
    return (gamma_H - gamma_L) * (1 - np.exp(-c * D2 / (D0 ** 2))) + gamma_L

def filtro_homomorfico(Y, gamma_H=1.5, gamma_L=0.25, D0=40, c=1.0, eps=1e-6):
    # Aplica filtrado homomórfico en el dominio espectral sobre la luminancia Y.
    N, M = Y.shape
    z = np.log(Y + eps)
    z_pad = np.pad(z, ((0, N), (0, M)), mode='symmetric')
    H = H_homomorfico(z_pad.shape, gamma_H, gamma_L, D0, c)
    S = H * np.fft.fftshift(np.fft.fft2(z_pad))
    s = np.fft.ifft2(np.fft.ifftshift(S)).real[:N, :M]
    Y_out = np.exp(s)
    return reescalar(Y_out, 0.5, 99.5)

def durand_hdr(E, sigma_s=None, sigma_r=0.4, contraste=30):
    # Compresión HDR de Durand basada en descomposición base/detalle con filtro bilateral.
    if E.ndim == 3:
        Y = luminancia(E)
    else:
        Y = E.copy()
        
    logY = np.log10(Y + 1e-12).astype(np.float32)
    if sigma_s is None:
        sigma_s = 0.02 * max(Y.shape)
        
    base = cv2.bilateralFilter(logY, d=-1, sigmaColor=float(sigma_r), sigmaSpace=float(sigma_s))
    detalle = logY - base
    
    b_min, b_max = base.min(), base.max()
    if b_max > b_min:
        k = np.log10(contraste) / (b_max - b_min)
    else:
        k = 1.0
        
    logY_out = k * (base - b_max) + detalle
    Y_out = 10 ** logY_out
    return reescalar(Y_out, 0.5, 99.5)

def compresion_rgb_independiente(E, metodo='homomorfico', **kwargs):
    # Aplica la compresión HDR de forma independiente sobre cada canal R, G y B.
    canales = []
    for c in range(3):
        ch = E[..., c]
        if metodo == 'homomorfico':
            ch_out = filtro_homomorfico(ch, **kwargs)
        else:
            ch_out = durand_hdr(ch, **kwargs)
        canales.append(ch_out)
    return np.stack(canales, axis=-1)

def a_pantalla(x, gamma=2.2):
    # Corrección gamma para desplegar imágenes lineales en [0, 1].
    return np.clip(x, 0.0, 1.0) ** (1.0 / gamma)

def reconstruct_color_ratio(E, Y, Y_prima, beta=1.0, eps=1e-6):
    # Estrategia 1: Reconstrucción de color C' = Y' * (C / (Y + eps))^beta
    ratio = (E / (Y[..., None] + eps)) ** beta
    E_out = Y_prima[..., None] * ratio
    return a_pantalla(np.clip(E_out, 0.0, 1.0))

def reconstruct_color_hsv(E, Y_prime, s_factor=1.0, Y=None, eps=1e-6):
    if Y is None:
        Y = luminancia(E)
        
    # Cromaticidad normalizada limpia en [0, 1]
    chroma_rgb = np.clip(E / (Y[..., None] + eps), 0.0, 1.0)
    chroma_u8 = (chroma_rgb * 255.0).astype(np.uint8)
    
    hsv = cv2.cvtColor(chroma_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_factor, 0.0, 255.0)
    hsv[..., 2] = np.clip(Y_prime * 255.0, 0.0, 255.0)
    
    rgb_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0
    return a_pantalla(np.clip(rgb_out, 0.0, 1.0))

# Métricas de color
def calculate_color_metrics(E_orig, E_proc):
    orig_norm = reescalar(E_orig, 0.5, 99.5)
    proc_norm = np.clip(E_proc, 0.0, 1.0)
    
    orig_u8 = (orig_norm * 255.0).astype(np.uint8)
    proc_u8 = (proc_norm * 255.0).astype(np.uint8)
    
    # Croma C* en Lab
    lab_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_proc = cv2.cvtColor(proc_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    chroma_orig = float(np.mean(np.sqrt(lab_orig[..., 1]**2 + lab_orig[..., 2]**2)))
    chroma_proc = float(np.mean(np.sqrt(lab_proc[..., 1]**2 + lab_proc[..., 2]**2)))
    
    # Tono H en HSV
    hsv_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_proc = cv2.cvtColor(proc_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Diferencia angular en grados (0..360)
    h_diff = np.abs(hsv_orig[..., 0] - hsv_proc[..., 0]) * 2.0
    h_diff = np.minimum(h_diff, 360.0 - h_diff)
    mean_hue_shift = float(np.mean(h_diff))
    
    return {
        'chroma_orig': round(chroma_orig, 2), # retorna el valor de croma promedio del original
        'chroma_proc': round(chroma_proc, 2), # retorna el valor de croma promedio del procesado
        'hue_shift_deg': round(mean_hue_shift, 2) # retorna el cambio medio de tono en grados
    }
