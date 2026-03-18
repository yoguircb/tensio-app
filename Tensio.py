# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:13:06 2026

@author: Rubén Castañeda Balderas
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.stats import linregress
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import streamlit.components.v1 as components


# ==========================================
# 0. CONFIGURACIÓN INICIAL Y DICCIONARIO
# ==========================================
st.set_page_config(page_title="Tensio", layout="wide", page_icon="⚙️")

# --- CONTROL DE ESTADO ---
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'es'
if 'resultados_cache' not in st.session_state:
    st.session_state['resultados_cache'] = {}

# --- FUNCIÓN DE RESETEO TOTAL (NUEVA Y CORREGIDA) ---
# Limpia todo el entorno (cálculos y selecciones de la interfaz) 
# al cambiar de archivo, pero respeta el idioma.
def on_file_change():
    idioma_actual = st.session_state.get('lang', 'es')
    st.session_state.clear()
    st.session_state['lang'] = idioma_actual
    st.session_state['resultados_cache'] = {}

# --- DICCIONARIO DE TRADUCCIONES UI Y EXCEL ---
TRAD = {
    'es': {
        'nav_title': '### 🧭 Navegación',
        'go_to': 'Ir a:',
        'data_proc': 'Procesador de Datos',
        'user_manual': 'Manual de Usuario',
        'about': '### 👨‍💻 Acerca del Desarrollo',
        'warn_title': '### ⚠️ Aviso de Responsabilidad',
        'upload': 'Sube tu archivo .raw de Instron',
        'sel_samples': '1. Selección de Muestras',
        'geom': '2. Geometría',
        'props': '3. Propiedades (con Unidades)',
        'map': '4. Mapeo',
        'stress_col': 'Columna Esfuerzo (Y):',
        'strain_col': 'Columna Deformación (X):',
        'strain_pct': '¿Deformación en %?',
        'inspect': '🔍 Inspección y Ajuste Individual',
        'sel_prob': 'Seleccionar Probeta:',
        'calc_status': '✅ {}: Calculada.',
        'recalc': '🔄 Recalcular',
        'sug_loaded': '⚠️ Sugerencia cargada. Por favor, revise y afine la Zona Elástica si es necesario antes de calcular.',
        'refine': 'Refinar Zona Elástica (Sugerencia Automática):',
        'calc_save': 'Calcular y Guardar',
        'inv_range': 'Rango inválido. Intenta seleccionar más puntos.',
        'results': '📊 Tabla de Resultados Acumulados',
        'progress': 'Progreso: {0} de {1} probetas.',
        'complete_msg': '¡Análisis completado! Generando Excel...',
        'download': '📥 Descargar Reporte Excel (.xlsx)',
        'warn_calc': 'Completa el cálculo para descargar.',
        'no_results': 'Aún no hay resultados.',
        
        # Textos para Gráficas, Tablas y Excel
        'Esfuerzo Máximo (MPa)': 'Esfuerzo Máximo (MPa)',
        'Deformación Máxima (%)': 'Deformación Máxima (%)',
        'Carga Máxima (N)': 'Carga Máxima (N)',
        'Módulo de Young (GPa)': 'Módulo de Young (GPa)',
        'Límite de Cedencia 0.2% (MPa)': 'Límite de Cedencia 0.2% (MPa)',
        'Límite de Proporcionalidad (Curvatura)': 'Límite de Proporcionalidad (Curvatura)',
        'r2_fit': 'R² Ajuste',
        'uncert_e': 'Incertidumbre E (± GPa)',
        'id_sample': 'ID_Muestra',
        'load_n': 'Carga (N)',
        'ext_mm': 'Extensión (mm)',
        'stress_mpa': 'Esfuerzo (MPa)',
        'strain_mm_mm': 'Deformación (mm/mm)',
        'strain_ax': 'Deformación (mm/mm)',
        'stress_ax': 'Esfuerzo (MPa)',
        'corrected_curve': 'Curva Corregida',
        'yield_02': 'Cedencia 0.2%',
        'prop_limit': 'Lím. Proporcionalidad',
        'full_curve': 'Curva Completa',
        'active_zone': 'Zona a Evaluar',
        'chart_title': 'Comparativa de Curvas Corregidas',
        'summary_sheet': 'Resumen',
        'result_title': 'Resultado'
    },
    'en': {
        'nav_title': '### 🧭 Navigation',
        'go_to': 'Go to:',
        'data_proc': 'Data Processor',
        'user_manual': 'User Manual',
        'about': '### 👨‍💻 About the Development',
        'warn_title': '### ⚠️ Disclaimer',
        'upload': 'Upload your Instron .raw file',
        'sel_samples': '1. Sample Selection',
        'geom': '2. Geometry',
        'props': '3. Properties (with Units)',
        'map': '4. Mapping',
        'stress_col': 'Stress Column (Y):',
        'strain_col': 'Strain Column (X):',
        'strain_pct': 'Strain in %?',
        'inspect': '🔍 Individual Inspection and Adjustment',
        'sel_prob': 'Select Specimen:',
        'calc_status': '✅ {}: Calculated.',
        'recalc': '🔄 Recalculate',
        'sug_loaded': '⚠️ Suggestion loaded. Please review and fine-tune the Elastic Zone if necessary before calculating.',
        'refine': 'Refine Elastic Zone (Automatic Suggestion):',
        'calc_save': 'Calculate and Save',
        'inv_range': 'Invalid range. Try selecting more points.',
        'results': '📊 Accumulated Results Table',
        'progress': 'Progress: {0} of {1} specimens.',
        'complete_msg': 'Analysis completed! Generating Excel...',
        'download': '📥 Download Excel Report (.xlsx)',
        'warn_calc': 'Complete the calculation to download.',
        'no_results': 'No results yet.',
        
        # Textos para Gráficas, Tablas y Excel
        'Esfuerzo Máximo (MPa)': 'Maximum Stress (MPa)',
        'Deformación Máxima (%)': 'Maximum Strain (%)',
        'Carga Máxima (N)': 'Maximum Load (N)',
        'Módulo de Young (GPa)': "Young's Modulus (GPa)",
        'Límite de Cedencia 0.2% (MPa)': 'Yield Strength 0.2% (MPa)',
        'Límite de Proporcionalidad (Curvatura)': 'Proportional Limit (Curvature)',
        'r2_fit': 'R² Fit',
        'uncert_e': 'Uncertainty E (± GPa)',
        'id_sample': 'Sample_ID',
        'load_n': 'Load (N)',
        'ext_mm': 'Extension (mm)',
        'stress_mpa': 'Stress (MPa)',
        'strain_mm_mm': 'Strain (mm/mm)',
        'strain_ax': 'Strain (mm/mm)',
        'stress_ax': 'Stress (MPa)',
        'corrected_curve': 'Corrected Curve',
        'yield_02': 'Yield 0.2%',
        'prop_limit': 'Prop. Limit',
        'full_curve': 'Full Curve',
        'active_zone': 'Active Zone',
        'chart_title': 'Corrected Curves Comparison',
        'summary_sheet': 'Summary',
        'result_title': 'Result'
    }
}

def t(key):
    """Retorna la traducción según el idioma seleccionado."""
    return TRAD[st.session_state['lang']].get(key, key)


# ==========================================
# 0.5. BARRA LATERAL (AUTORÍA Y LEGALES)
# ==========================================
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("<p style='text-align: center; font-size: 0.8em;'>Advanced Tensile Data Processing</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🌐 Idioma / Language")
    idx_lang = 0 if st.session_state['lang'] == 'es' else 1
    idioma_sel = st.radio("Sel:", ["Español", "English"], index=idx_lang, horizontal=True, label_visibility="collapsed")
    st.session_state['lang'] = 'es' if idioma_sel == "Español" else 'en'
    st.divider()
    
    st.markdown(t('nav_title'))
    opcion = st.radio(t('go_to'), [t('data_proc'), t('user_manual')])
    st.divider()
    
    st.markdown(t('about'))
    st.markdown("**Autor / Author:** M.S.C. Rubén Castañeda Balderas")
    st.markdown("Centro de Investigación en Materiales Avanzados S.C.")
    st.markdown("**Versión:** 1.3.6.2")
    st.divider() 
    
    st.info(
        "**Cita sugerida / Suggested citation:**\n\n"
        "*Castañeda, Rubén. (2026). TENSIO: Advanced Tensile Data Processing (Versión 1.3.6.2).*"
    )
    st.divider()
    
    st.markdown(t('warn_title'))
    st.caption(
        "Este software es una herramienta asistida. El autor no garantiza la "
        "exactitud absoluta debido a la variabilidad del ruido en archivos RAW. / "
        "This software is an assisted tool. The author does not guarantee absolute "
        "accuracy due to noise variability in RAW files."
    )


# ==========================================
# SECCIÓN DEL MANUAL DE USUARIO
# ==========================================
if opcion == t('user_manual'):
    st.title(f"📖 {t('user_manual')}")
    flip_url = "https://online.fliphtml5.com/swdyw/Tensio/" 
    components.iframe(flip_url, height=850, scrolling=False)
    try:
        with open("Tensio.pdf", "rb") as f:
            st.download_button(
                label="📥 Descargar Manual en PDF / Download PDF Manual",
                data=f,
                file_name="Manual_Tensio_CIMAV.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        pass 
    st.stop()

# ==========================================
# 1. FUNCIONES MATEMÁTICAS MEJORADAS (SIN CAMBIOS)
# ==========================================

def aplicar_filtro(Y, window_length=15, polyorder=3):
    if len(Y) < window_length:
        window_length = len(Y) if len(Y) % 2 != 0 else len(Y) - 1
    if window_length <= polyorder:
        return Y
    return savgol_filter(Y, window_length, polyorder)

def sugerir_rango_elastico(X, Y_raw):
    if len(Y_raw) < 10: return 0.0, 0.01
    Y = aplicar_filtro(Y_raw)
    idx_max = np.argmax(Y)
    if idx_max < 10: return 0.0, max(0.01, float(X[-1]) if len(X)>0 else 0.01)

    umbral_busqueda = Y[idx_max] * 0.35
    idx_search_limit = np.argmax(Y > umbral_busqueda)
    if idx_search_limit < 10: 
        idx_search_limit = min(idx_max, int(len(Y)*0.15))

    X_s = X[:idx_search_limit]
    Y_s = Y[:idx_search_limit]

    w_size = max(8, int(len(X_s) * 0.15)) 
    if w_size > len(X_s) // 2: w_size = max(5, len(X_s) // 2)

    max_metric = -np.inf
    best_slope = 0
    seed_s = 0
    seed_e = w_size

    for i in range(len(X_s) - w_size):
        x_win = X_s[i:i+w_size]
        y_win = Y_s[i:i+w_size]
        
        if len(x_win) < 3 or (x_win[-1] - x_win[0] == 0): continue
        slope, _, r_val, _, _ = linregress(x_win, y_win)
        r2 = r_val**2
        
        if r2 > 0.98:
            metric = slope * (r2**5)
            if metric > max_metric:
                max_metric = metric
                best_slope = slope
                seed_s = i
                seed_e = i + w_size

    if max_metric == -np.inf:
        for i in range(len(X_s) - w_size):
            x_win = X_s[i:i+w_size]
            y_win = Y_s[i:i+w_size]
            if x_win[-1] - x_win[0] == 0: continue
            slope, _, r_val, _, _ = linregress(x_win, y_win)
            r2 = r_val**2
            if r2 > 0.95:
                metric = slope * (r2**3)
                if metric > max_metric:
                    max_metric = metric
                    best_slope = slope
                    seed_s = i
                    seed_e = i + w_size

    if max_metric == -np.inf: 
        return 0.0, float(X_s[-1] if len(X_s)>0 else 0.01)

    idx_s = seed_s
    idx_e = seed_e
    step = max(1, len(X) // 200)

    if best_slope > 10000:
        while idx_e < idx_max - 1:
            nxt = min(idx_e + step, idx_max - 1)
            m_new, _, r_new, _, _ = linregress(X[idx_s:nxt], Y[idx_s:nxt])
            if r_new**2 > 0.980 and m_new > best_slope * 0.95: idx_e = nxt
            else: break
                
        while idx_s > 0:
            prv = max(idx_s - step, 0)
            m_new, _, r_new, _, _ = linregress(X[prv:idx_e], Y[prv:idx_e])
            if r_new**2 > 0.970 and m_new > best_slope * 0.90: idx_s = prv
            else: break
    else:
        while idx_e < idx_max - 1:
            nxt = min(idx_e + step, idx_max - 1)
            m_new, _, r_new, _, _ = linregress(X[idx_s:nxt], Y[idx_s:nxt])
            if r_new**2 > 0.960 and m_new > best_slope * 0.85: idx_e = nxt
            else: break
                
        while idx_s > 0:
            prv = max(idx_s - step, 0)
            m_new, _, r_new, _, _ = linregress(X[prv:idx_e], Y[prv:idx_e])
            if r_new**2 > 0.950 and m_new > best_slope * 0.80: idx_s = prv
            else: break

    if idx_e <= idx_s or (X[idx_e] - X[idx_s]) <= 0:
        idx_e = min(idx_s + 5, len(X)-1)

    return float(X[idx_s]), float(X[idx_e])

def calcular_propiedades_finales(X_raw, Y_raw, rango_seleccionado, props_sel):
    Y_smooth = aplicar_filtro(Y_raw)
    start_val, end_val = rango_seleccionado
    
    mask_trunc = (X_raw >= start_val)
    X_trunc = X_raw[mask_trunc]
    Y_trunc = Y_smooth[mask_trunc]
    
    mask_reg = (X_trunc >= start_val) & (X_trunc <= end_val)
    if np.sum(mask_reg) <= 2: return None, None, None, None

    slope, intercept, r_val, p_val, std_err = linregress(X_trunc[mask_reg], Y_trunc[mask_reg])
    
    x_zero = -intercept / slope if slope != 0 else start_val
    X_translated = X_trunc - x_zero
    Y_translated = Y_trunc 
    
    mask_positive = X_translated > 0
    X_clean = X_translated[mask_positive]
    Y_clean = Y_translated[mask_positive]
    
    X_final = np.insert(X_clean, 0, 0.0)
    Y_final = np.insert(Y_clean, 0, 0.0)
    
    if len(X_final) < 5: return None, None, None, None

    res = {}
    if "Esfuerzo Máximo (MPa)" in props_sel: res["Esfuerzo Máximo (MPa)"] = np.max(Y_final)
    
    max_def = np.max(X_final)
    if "Deformación Máxima (%)" in props_sel: 
        res["Deformación Máxima (%)"] = max_def * 100 

    if "Módulo de Young (GPa)" in props_sel: 
        res["Módulo de Young (GPa)"] = slope / 1000.0
        res["R^2 Ajuste"] = r_val**2
        res["Incertidumbre E (± GPa)"] = std_err / 1000.0 
    
    if "Límite de Cedencia 0.2% (MPa)" in props_sel:
        if slope > 0:
            offset_strain = 0.002
            y_line_offset = slope * (X_final - offset_strain)
            diff = Y_final - y_line_offset
            search_indices = np.where(X_final > (offset_strain * 0.5))[0]
            cedencia_val = None
            if len(search_indices) > 0:
                start_idx = search_indices[0]
                for i in range(start_idx, len(diff) - 1):
                    if diff[i] >= 0 and diff[i+1] < 0:
                        x1, y1 = X_final[i], Y_final[i]
                        x2, y2 = X_final[i+1], Y_final[i+1]
                        m_c = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                        if (m_c - slope) != 0:
                            x_yield = (m_c * x1 - y1 - slope * offset_strain) / (m_c - slope)
                            cedencia_val = slope * (x_yield - offset_strain)
                        else:
                            cedencia_val = y1
                        break
            res["Límite de Cedencia 0.2% (MPa)"] = cedencia_val

    if "Límite de Proporcionalidad (Curvatura)" in props_sel:
        if slope > 0:
            dX = np.gradient(X_final)
            dY = np.gradient(Y_final)
            mask_dx = dX > 1e-8
            Et = np.zeros_like(Y_final)
            Et[mask_dx] = dY[mask_dx] / dX[mask_dx]
            
            w_len = max(5, len(Et)//20)
            if w_len % 2 == 0: w_len += 1
            if w_len > 3: Et_smooth = aplicar_filtro(Et, window_length=w_len, polyorder=2)
            else: Et_smooth = Et
            
            umbral_caida = slope * 0.75 
            idx_seguros = np.where(X_final > (end_val * 0.5))[0]
            if len(idx_seguros) > 0:
                start_i = idx_seguros[0]
                caidas = np.where(Et_smooth[start_i:] < umbral_caida)[0]
                if len(caidas) > 0:
                    idx_prop = start_i + caidas[0]
                    res["Límite de Proporcionalidad (Curvatura)"] = Y_final[idx_prop]
                else:
                    res["Límite de Proporcionalidad (Curvatura)"] = None

    return res, X_final, Y_final, slope

# ==========================================
# 2. INTERFAZ Y CARGA
# ==========================================
st.title("TENSIO: Advanced Tensile Data Processing")

# --- CONEXIÓN DE LA FUNCIÓN DE RESETEO TOTAL ---
uploaded_file = st.file_uploader(t('upload'), type=['raw', 'csv', 'txt'], on_change=on_file_change)

if not uploaded_file: st.stop()

try: content = uploaded_file.getvalue().decode('latin-1')
except: 
    st.error("Error codificación UTF-8. Intenta guardar tu archivo como UTF-8 o ANSI.")
    st.stop()

buffer = io.StringIO(content)
lineas = buffer.readlines()

lista_probetas = []
meta_temp = {}
data_temp = []
leyendo_datos = False
en_bloque_valido = False 

for linea in lineas:
    linea_limpia = linea.strip()
    parts = linea_limpia.replace('"', '').split(',')
    
    if len(parts) >= 2:
        key = parts[0].strip()
        val = parts[1].strip()
        if key == "Probeta":
            if en_bloque_valido:
                meta_temp["__RAW_DATA__"] = "".join(data_temp)
                lista_probetas.append(meta_temp.copy())
            en_bloque_valido = True
            meta_temp = {}
            data_temp = []
            leyendo_datos = False
            meta_temp["ID_Muestra"] = f"Probeta {val}"
            continue 
    if not en_bloque_valido: continue
    if ("Tiempo" in linea_limpia or "Time" in linea_limpia) and not leyendo_datos:
        leyendo_datos = True
        data_temp.append(linea) 
        continue
    if leyendo_datos:
        if parts[0] and parts[0][0].isalpha() and "Probeta" not in parts[0]: leyendo_datos = False
        else: data_temp.append(linea); continue
    if not leyendo_datos and len(parts) >= 2:
        key = parts[0].strip()
        val = parts[1].strip()
        if len(parts) > 2 and parts[2].strip(): key = f"{key} ({parts[2].strip()})"
        if key not in ["Tiempo", "Time", "Carga", "Extensión", "Load"]:
            try: meta_temp[key] = float(val)
            except: meta_temp[key] = val

if en_bloque_valido and meta_temp:
    meta_temp["__RAW_DATA__"] = "".join(data_temp)
    lista_probetas.append(meta_temp.copy())

df_meta = pd.DataFrame(lista_probetas)

if '__RAW_DATA__' not in df_meta.columns or df_meta.empty:
    df_meta = pd.DataFrame({
        'ID_Muestra': ['Probeta_Validacion'],
        '__RAW_DATA__': [content]
    })

cols_orden = ['ID_Muestra'] + [c for c in df_meta.columns if c != 'ID_Muestra' and c != '__RAW_DATA__']
cols_existentes = [col for col in cols_orden if col in df_meta.columns]
if '__RAW_DATA__' in df_meta.columns:
    cols_existentes.append('__RAW_DATA__')
df_meta = df_meta[cols_existentes]

st.success(f"✅ {len(df_meta)} probetas cargadas / loaded.")
st.divider()

# ==========================================
# 3. SELECTORES
# ==========================================
st.subheader(t('sel_samples'))
probetas_a_procesar = []

if 'ID_Muestra' not in df_meta.columns:
    if df_meta.empty: df_meta = pd.DataFrame({'ID_Muestra': ['Probeta_Validacion']})
    else: df_meta['ID_Muestra'] = [f"Probeta {i+1}" for i in range(len(df_meta))]

ids = df_meta['ID_Muestra'].tolist()

with st.container():
    cols_p = st.columns(4)
    for i, pid in enumerate(ids):
        label = f"✅ {pid}" if pid in st.session_state['resultados_cache'] else pid
        if cols_p[i % 4].checkbox(label, value=True, key=f"p_{i}"):
            probetas_a_procesar.append(pid)
st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader(t('geom'))
    blacklist = ["__RAW_DATA__", "ID_Muestra", "Entrada de texto", "Nota", "Método", "Ciclado", "Entrada num", "Peso", "Densidad", "Área Final", "Relación", "Separación", "fijación", "Etiqueta", "Tipo de ensayo", "Usuario", "Cliente", "Muestra", "Nombre"]
    cols_limpias = [c for c in df_meta.columns if not any(bad in c for bad in blacklist)]
    defaults_keywords = ["Anchura", "Espesor", "Diámetro", "Area", "Área", "Longitud", "Geometría"]
    meta_sel = []
    if cols_limpias:
        grid_geo = st.columns(2)
        for i, col in enumerate(cols_limpias):
            checked = any(kw in col for kw in defaults_keywords)
            if grid_geo[i % 2].checkbox(col, value=checked, key=f"m_{col}"): meta_sel.append(col)

with c2:
    st.subheader(t('props'))
    # Mapeo interno-externo con diccionario de traducción en tiempo real
    opciones_props_keys = [
        "Esfuerzo Máximo (MPa)", 
        "Deformación Máxima (%)", 
        "Carga Máxima (N)", 
        "Módulo de Young (GPa)", 
        "Límite de Cedencia 0.2% (MPa)",
        "Límite de Proporcionalidad (Curvatura)"
    ]
    props_sel = []
    for prop_key in opciones_props_keys:
        if st.checkbox(t(prop_key), value=True, key=f"pr_{prop_key}"): props_sel.append(prop_key)

st.divider()
st.subheader(t('map'))
row_dummy = df_meta.iloc[0]
df_dummy = pd.read_csv(io.StringIO(row_dummy['__RAW_DATA__']))
headers_datos = [c.strip().replace('"', '') for c in df_dummy.columns]

idx_stress = next((i for i, h in enumerate(headers_datos) if "Esfuerzo" in h or "Stress" in h), 0)
idx_strain = next((i for i, h in enumerate(headers_datos) if "Deformación" in h or "Strain" in h), 0)

c_m1, c_m2, c_m3 = st.columns(3)
col_y_name = c_m1.selectbox(t('stress_col'), headers_datos, index=idx_stress)
col_x_name = c_m2.selectbox(t('strain_col'), headers_datos, index=idx_strain)
es_porcentaje = c_m3.checkbox(t('strain_pct'), value=True)

# ==========================================
# 4. CÁLCULO
# ==========================================
st.divider()
st.subheader(t('inspect'))

if probetas_a_procesar:
    col_sel_prob, col_status = st.columns([1, 2])
    with col_sel_prob:
        probeta_visual = st.selectbox(t('sel_prob'), probetas_a_procesar)
    
    esta_calculada = probeta_visual in st.session_state['resultados_cache']
    
    if esta_calculada:
        st.success(t('calc_status').format(probeta_visual))
        if st.button(t('recalc')):
            del st.session_state['resultados_cache'][probeta_visual]
            st.rerun()
            
        datos_corr = st.session_state['resultados_cache'][probeta_visual]
        X_plot = datos_corr['X_final']
        Y_plot = datos_corr['Y_final']
        slope_f = datos_corr['slope']
        
        y_max_real = np.max(Y_plot)
        y_techo = y_max_real * 1.1
        x_limit = y_techo / slope_f if slope_f != 0 else np.max(X_plot)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_plot, y=Y_plot, mode='lines', name=t('corrected_curve'), line=dict(color='#2ca02c', width=3)))
        
        x_line = np.linspace(0, x_limit, 50)
        y_young = slope_f * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_young, mode='lines', name=f"Young {slope_f/1000:.1f} GPa", line=dict(color='orange', dash='dash')))
        
        offset = 0.002
        y_offset = slope_f * (x_line - offset)
        mask_off = (y_offset >= 0) & (y_offset <= y_techo)
        fig.add_trace(go.Scatter(x=x_line[mask_off], y=y_offset[mask_off], mode='lines', name=t('yield_02'), line=dict(color='red', dash='dot')))
        
        if "Límite de Proporcionalidad (Curvatura)" in datos_corr['resultados']:
            val_curvatura = datos_corr['resultados']["Límite de Proporcionalidad (Curvatura)"]
            if val_curvatura is not None:
                fig.add_hline(y=val_curvatura, line_dash="dash", line_color="purple", annotation_text=t('prop_limit'))

        fig.update_layout(title=f"{t('result_title')}: {probeta_visual}", xaxis_title=t('strain_ax'), yaxis_title=t('stress_ax'), template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        row_viz = df_meta[df_meta['ID_Muestra'] == probeta_visual].iloc[0]
        df_viz = pd.read_csv(io.StringIO(row_viz['__RAW_DATA__']))
        df_viz.columns = [c.strip().replace('"', '') for c in df_viz.columns]
        
        Y_raw = pd.to_numeric(df_viz[col_y_name], errors='coerce').fillna(0).values
        X_raw = pd.to_numeric(df_viz[col_x_name], errors='coerce').fillna(0).values
        if es_porcentaje: X_raw = X_raw / 100.0
        
        auto_range = sugerir_rango_elastico(X_raw, Y_raw)
        
        st.info(t('sug_loaded'))
        
        max_x = float(X_raw.max()) if len(X_raw) > 0 else 0.1
        limite_superior = max_x / 3
        
        val_start = min(auto_range[0], limite_superior)
        val_end = min(auto_range[1], limite_superior)
        if val_start >= val_end: 
            val_end = min(val_start + 0.001, limite_superior)

        rango_final = st.slider(
            t('refine'), 
            0.0, float(limite_superior), 
            value=(float(val_start), float(val_end)), 
            step=0.0001, 
            format="%.4f"
        )
        
        fig = go.Figure()
        Y_smooth_plot = aplicar_filtro(Y_raw)
        
        fig.add_trace(go.Scatter(x=X_raw, y=Y_smooth_plot, mode='lines', name=t('full_curve'), line=dict(color='lightgray')))
        mask_man = (X_raw >= rango_final[0]) & (X_raw <= rango_final[1])
        fig.add_trace(go.Scatter(x=X_raw[mask_man], y=Y_smooth_plot[mask_man], mode='lines', name=t('active_zone'), line=dict(color='#2ca02c', width=4)))
        
        zoom_max = rango_final[1] * 2.5
        fig.update_layout(xaxis_range=[0, zoom_max], yaxis_range=[0, np.max(Y_raw)*0.8], xaxis_title=t('strain_ax'), yaxis_title=t('stress_ax'))

        st.plotly_chart(fig, use_container_width=True)
        
        if st.button(t('calc_save'), type="primary"):
            res_calc, X_f, Y_f, m_f = calcular_propiedades_finales(X_raw, Y_raw, rango_final, props_sel)
            
            if res_calc:
                if "Carga Máxima (N)" in props_sel:
                    df_viz = pd.read_csv(io.StringIO(row_viz['__RAW_DATA__']))
                    df_viz.columns = [c.strip().replace('"', '') for c in df_viz.columns]
                    col_load = next((c for c in df_viz.columns if "Carga" in c or "Load" in c), None)
                    res_calc["Carga Máxima (N)"] = df_viz[col_load].max() if col_load else 0

                res_calc["ID_Muestra"] = probeta_visual
                
                if "Deformación Máxima (%)" in res_calc and not es_porcentaje:
                     res_calc["Deformación Máxima (%)"] = res_calc["Deformación Máxima (%)"] 
                
                max_stress = np.max(Y_raw) if np.max(Y_raw) > 0 else 1
                df_temp = pd.read_csv(io.StringIO(row_viz['__RAW_DATA__']))
                df_temp.columns = [c.strip().replace('"', '') for c in df_temp.columns]
                c_load = next((c for c in df_temp.columns if "Carga" in c or "Load" in c), None)
                c_ext = next((c for c in df_temp.columns if "Exten" in c or "Displa" in c), None)
                
                factor_area = 1.0
                factor_len = 1.0
                if c_load: factor_area = df_temp[c_load].max() / max_stress
                max_x_raw = np.max(X_raw) if np.max(X_raw) > 0 else 1
                if c_ext: factor_len = df_temp[c_ext].max() / max_x_raw

                st.session_state['resultados_cache'][probeta_visual] = {
                    'resultados': res_calc,
                    'X_final': X_f, 
                    'Y_final': Y_f, 
                    'slope': m_f,
                    'factor_area': factor_area,
                    'factor_len': factor_len
                }
                st.rerun()
            else:
                st.error(t('inv_range'))

# ==========================================
# 5. RESULTADOS Y EXCEL
# ==========================================
st.divider()
st.subheader(t('results'))

ids_calculados = list(st.session_state['resultados_cache'].keys())

if ids_calculados:
    lista_res = [st.session_state['resultados_cache'][k]['resultados'] for k in ids_calculados]
    df_resultados = pd.DataFrame(lista_res)
    
    cols_geo_validas = ['ID_Muestra'] + [c for c in meta_sel if c in df_meta.columns]
    df_geo_final = df_meta[df_meta['ID_Muestra'].isin(ids_calculados)][cols_geo_validas]
    
    df_final_show = pd.merge(df_geo_final, df_resultados, on="ID_Muestra")
    
    # --- TRADUCCIÓN COMPLETA DE ENCABEZADOS DE LA TABLA ---
    rename_dict = {
        "ID_Muestra": t('id_sample'),
        "Esfuerzo Máximo (MPa)": t('Esfuerzo Máximo (MPa)'),
        "Deformación Máxima (%)": t('Deformación Máxima (%)'),
        "Carga Máxima (N)": t('Carga Máxima (N)'),
        "Módulo de Young (GPa)": t('Módulo de Young (GPa)'),
        "Límite de Cedencia 0.2% (MPa)": t('Límite de Cedencia 0.2% (MPa)'),
        "Límite de Proporcionalidad (Curvatura)": t('Límite de Proporcionalidad (Curvatura)'),
        "R^2 Ajuste": t('r2_fit'),
        "Incertidumbre E (± GPa)": t('uncert_e')
    }
    df_display = df_final_show.rename(columns=rename_dict)
    
    st.dataframe(df_display, use_container_width=True)
    
    total_objetivo = len(probetas_a_procesar)
    calculadas_validas = [pid for pid in ids_calculados if pid in probetas_a_procesar]
    total_listas = len(calculadas_validas)
    
    progreso_val = total_listas / total_objetivo if total_objetivo > 0 else 0
    st.write(t('progress').format(total_listas, total_objetivo))
    st.progress(progreso_val)
    
    if total_listas == total_objetivo and total_objetivo > 0:
        st.success(t('complete_msg'))
        
        buffer_excel = io.BytesIO()
        
        try:
            import xlsxwriter
            engine_str = 'xlsxwriter'
            has_xlsxwriter = True
        except ImportError:
            engine_str = 'openpyxl' 
            has_xlsxwriter = False
            st.warning("⚠️ Librería 'xlsxwriter' no detectada. Se generará el Excel SIN gráficos.")

        with pd.ExcelWriter(buffer_excel, engine=engine_str) as writer:
            # Hoja de Resumen Traducida
            nombre_hoja_resumen = t('summary_sheet')
            df_display.to_excel(writer, sheet_name=nombre_hoja_resumen, index=False)
            
            if has_xlsxwriter:
                workbook = writer.book
                ws_resumen = writer.sheets[nombre_hoja_resumen]
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
                chart.set_title({'name': t('chart_title')})
                chart.set_x_axis({'name': t('strain_ax')})
                chart.set_y_axis({'name': t('stress_ax')})
            
            colores = ['#FF0000', '#0000FF', '#008000', '#FFA500', '#800080', '#00CED1', '#FF1493', '#8B4513']
            
            for i, pid in enumerate(calculadas_validas):
                cache = st.session_state['resultados_cache'][pid]
                X_f = cache['X_final']
                Y_f = cache['Y_final']
                f_area = cache['factor_area']
                f_len = cache['factor_len']
                
                Load_corr = Y_f * f_area
                Ext_corr = X_f * f_len
                
                # Columnas internas del Excel Traducidas
                df_ind = pd.DataFrame({
                    t('load_n'): Load_corr,
                    t('ext_mm'): Ext_corr,
                    t('stress_mpa'): Y_f,
                    t('strain_mm_mm'): X_f
                })
                
                safe_name = pid.replace("Probeta ", "P_").replace(":", "")[:30]
                df_ind.to_excel(writer, sheet_name=safe_name, index=False)
                
                if has_xlsxwriter:
                    n_rows = len(df_ind)
                    color_code = colores[i % len(colores)]
                    chart.add_series({
                        'name':       safe_name,
                        'categories': [safe_name, 1, 3, n_rows, 3], 
                        'values':     [safe_name, 1, 2, n_rows, 2], 
                        'line':       {'color': color_code, 'width': 1.5},
                        'marker':     {'type': 'none'}
                    })
            
            if has_xlsxwriter:
                ws_resumen.insert_chart('G2', chart, {'x_scale': 2, 'y_scale': 2})
            
        st.download_button(
            label=t('download'),
            data=buffer_excel.getvalue(),
            file_name="Reporte_Instron_Completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    elif total_objetivo > 0:
        st.warning(t('warn_calc'))

else:
    st.info(t('no_results'))