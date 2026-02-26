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
import plotly.graph_objects as go
import streamlit.components.v1 as components


# ==========================================
# 0. CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(page_title="Tensio", layout="wide", page_icon="⚙️")

# ==========================================
# 0.5. BARRA LATERAL (AUTORÍA Y LEGALES)
# ==========================================
with st.sidebar:
    # --- LOGOTIPO ---
    # Mostramos el logo centrado
    st.image("logo.png", use_container_width=True)
    
    st.markdown("<p style='text-align: center; font-size: 0.8em;'>Advanced Tensile Data Processing</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # === NUEVO: MENÚ DE NAVEGACIÓN ===
    st.markdown("### 🧭 Navegación")
    opcion = st.radio("Ir a:", ["Procesador de Datos", "Manual de Usuario"])
    st.divider()
    # ==================================
    
    st.markdown("### 👨‍💻 Acerca del Desarrollo")
    st.markdown("**Autor:** M.S.C. Rubén Castañeda Balderas")
    st.markdown("**Empresa:** Centro de Investigación en Materiales Avanzados S.C.")
    st.markdown("**Versión:** 1.0.0")
    
    st.divider() # Línea divisoria visual
    
    # --- SECCIÓN DE CITACIÓN ---
    st.info(
        "**Cita sugerida:**\n\n"
        "Si este software es utilizado para generar datos, gráficas o resultados "
        "destinados a publicaciones académicas, informes técnicos o tesis, se "
        "solicita citar de la siguiente manera:\n\n"
        "*Castañeda, Rubén. (2026). TENSIO: Advanced Tensile Data Processing (Versión 1.1.0).*"
    )
    
    st.divider()
    
    # --- AVISO LEGAL ---
    st.markdown("### ⚠️ Aviso de Responsabilidad")
    st.caption(
        "**Limitación de Responsabilidad:** Este software es una herramienta de procesamiento "
        "de datos asistida y se suministra 'as is' (tal como está). El autor no garantiza la "
        "exactitud absoluta de los cálculos de ingeniería derivados (Módulo de Young, "
        "Límites de Cedencia o Deformación) debido a la variabilidad inherente en el ruido "
        "de señal de los archivos RAW y la sensibilidad de los algoritmos de regresión lineal. "
        "\n\n"
        "**Validación Obligatoria:** Los resultados generados no constituyen una certificación "
        "de cumplimiento de normas (ASTM, ISO, etc.). Es responsabilidad imperativa del usuario "
        "verificar la correlación entre las curvas gráficas y los valores tabulados antes de su "
        "uso en diseños estructurales, procesos de manufactura o publicaciones científicas. "
        "\n\n"
        "**Uso de Datos:** El autor queda exento de cualquier responsabilidad por daños directos, "
        "indirectos o incidentales resultantes de fallos materiales o decisiones basadas en el "
        "output de esta aplicación."
    )

if 'resultados_cache' not in st.session_state:
    st.session_state['resultados_cache'] = {}

# ==========================================
# SECCIÓN DEL MANUAL DE USUARIO
# ==========================================
if opcion == "Manual de Usuario":
    st.title("📖 Manual de Usuario Interactivo")
    
    # Pega aquí tu link de FlipHTML5
    flip_url = "https://online.fliphtml5.com/swdyw/Tensio/" 
    
    # Mostrar el libro interactivo
    components.iframe(flip_url, height=850, scrolling=False)
    
    # Botón de descarga
    try:
        with open("Tensio.pdf", "rb") as f:
            st.download_button(
                label="📥 Descargar Manual en PDF",
                data=f,
                file_name="Manual_Tensio_CIMAV.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        pass # Si no encuentra el PDF, no hace nada
        
    st.stop() # ¡MAGIA! Esto evita que se cargue el resto de la app si estamos en el manual.

# ==========================================
        
# ==========================================
# 1. FUNCIONES MATEMÁTICAS MEJORADAS
# ==========================================
def sugerir_rango_elastico(X, Y):
    # Evitar índices vacíos
    if len(Y) == 0: return 0.0, 0.01

    idx_max = np.argmax(Y)
    if idx_max < 10: return 0.0, max(0.01, float(X[-1]) if len(X)>0 else 0.01)
    
    X_search = X[:idx_max]
    Y_search = Y[:idx_max]
    window_size = max(5, int(len(X_search) * 0.05)) 
    step = max(1, int(window_size / 2))
    
    best_r2 = -np.inf
    best_slope = -np.inf
    best_range = (0.0, 0.01)
    
    for i in range(0, len(X_search) - window_size, step):
        x_win = X_search[i : i + window_size]
        y_win = Y_search[i : i + window_size]
        if len(x_win) < 2 or (x_win[-1] - x_win[0] == 0): continue

        slope, intercept, r_val, _, _ = linregress(x_win, y_win)
        r2 = r_val**2
        
        if r2 > 0.995:
            if slope > best_slope:
                best_slope = slope
                best_r2 = r2
                best_range = (float(x_win[0]), float(x_win[-1]))
        elif best_slope == -np.inf and r2 > best_r2:
            best_r2 = r2
            best_range = (float(x_win[0]), float(x_win[-1]))
    return best_range

def calcular_propiedades_finales(X_raw, Y_raw, rango_seleccionado, props_sel):
    start_val, end_val = rango_seleccionado
    
    # 1. Regresión en zona seleccionada
    mask_reg = (X_raw >= start_val) & (X_raw <= end_val)
    if np.sum(mask_reg) <= 2: return None, None, None, None

    slope, intercept, r_val, _, _ = linregress(X_raw[mask_reg], Y_raw[mask_reg])
    
    # 2. Corrección de Origen (Toe Compensation)
    # x_zero es donde la línea de regresión cruza Y=0
    x_zero = -intercept / slope if slope != 0 else start_val
    
    # 3. Desplazamiento de datos
    X_shifted = X_raw - x_zero
    Y_shifted = Y_raw 
    
    # 4. LIMPIEZA ESTRICTA [CORRECCIÓN APLICADA AQUÍ]
    # Filtramos no solo por X positivo, sino eliminamos la 'cola' plana inicial
    # Nos quedamos con datos donde X > 0. Si hay ruido en Y cerca de cero, lo limpiamos.
    # Encontramos el índice donde X empieza a ser positivo consistentemente
    
    mask_valid = X_shifted > 0
    X_final = X_shifted[mask_valid]
    Y_final = Y_shifted[mask_valid]
    
    # Forzamos el inicio (0,0) estético
    X_final = np.insert(X_final, 0, 0.0)
    Y_final = np.insert(Y_final, 0, 0.0)
    
    # Si después de limpiar quedan muy pocos datos, abortar
    if len(X_final) < 5: return None, None, None, None

    res = {}
    
    if "Esfuerzo Máximo (MPa)" in props_sel: res["Esfuerzo Máximo (MPa)"] = np.max(Y_final)
    
    # Ajustar deformación máxima si es porcentaje
    max_def = np.max(X_final)
    if "Deformación Máxima (%)" in props_sel: 
        # Nota: X_final viene en mm/mm o % según la entrada, pero aquí asumimos consistencia
        # Se multiplicará/ajustará fuera si es necesario, o aquí mismo si sabemos la unidad base
        res["Deformación Máxima (%)"] = max_def * 100 # Asumiendo base unitaria para reporte

    if "Módulo de Young (GPa)" in props_sel: 
        res["Módulo de Young (GPa)"] = slope / 1000.0
        res["R^2 Ajuste"] = r_val**2
    
    if "Límite de Cedencia 0.2% (MPa)" in props_sel:
        if slope > 0:
            offset_strain = 0.002
            y_line_offset = slope * (X_final - offset_strain)
            diff = Y_final - y_line_offset
            
            # Buscamos cruce después del offset positivo
            # Solo buscar donde X > 0.002
            search_indices = np.where(X_final > (offset_strain * 1.1))[0]
            
            cedencia_val = None
            if len(search_indices) > 0:
                start_idx = search_indices[0]
                # Buscar donde la diferencia cambia de signo (la curva cruza hacia abajo la línea)
                # diff > 0 significa curva por encima de línea. diff < 0 curva por debajo.
                crossings = np.where(diff[start_idx:] < 0)[0]
                if crossings.size > 0:
                    cedencia_val = Y_final[start_idx + crossings[0]]
            
            res["Límite de Cedencia 0.2% (MPa)"] = cedencia_val

    return res, X_final, Y_final, slope

# ==========================================
# 2. INTERFAZ Y CARGA
# ==========================================
st.title("TENSIO: Advanced Tensile Data Processing")

uploaded_file = st.file_uploader("Sube tu archivo .raw de Instron", type=['raw', 'csv', 'txt'])
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
cols_orden = ['ID_Muestra'] + [c for c in df_meta.columns if c != 'ID_Muestra' and c != '__RAW_DATA__']
#ESTO LE VOY A CAMBIAR df_meta = df_meta[cols_orden + ['__RAW_DATA__']]
# --- CÓDIGO SEGURO ANTI-ERRORES ---
# Verificamos qué columnas realmente existen antes de intentar ordenarlas
cols_existentes = [col for col in cols_orden if col in df_meta.columns]
if '__RAW_DATA__' in df_meta.columns:
    cols_existentes.append('__RAW_DATA__')
df_meta = df_meta[cols_existentes]
# ----------------------------------
st.success(f"Archivo cargado: {len(df_meta)} probetas.")
st.divider()

# ==========================================
# 3. SELECTORES
# ==========================================
st.subheader("1. Selección de Muestras")
probetas_a_procesar = []
#OTRO CAMBIO SOLICITADO ids = df_meta['ID_Muestra'].tolist()
# --- CÓDIGO SEGURO ANTI-ERRORES PARA IDs ---
if 'ID_Muestra' not in df_meta.columns:
    # Si el archivo es un CSV puro sin metadata, creamos una probeta genérica
    if df_meta.empty:
        df_meta = pd.DataFrame({'ID_Muestra': ['Probeta_Validacion']})
    else:
        df_meta['ID_Muestra'] = [f"Probeta {i+1}" for i in range(len(df_meta))]

ids = df_meta['ID_Muestra'].tolist()
# -------------------------------------------
with st.container():
    cols_p = st.columns(4)
    for i, pid in enumerate(ids):
        label = f"✅ {pid}" if pid in st.session_state['resultados_cache'] else pid
        if cols_p[i % 4].checkbox(label, value=True, key=f"p_{i}"):
            probetas_a_procesar.append(pid)
st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("2. Geometría")
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
    st.subheader("3. Propiedades (con Unidades)")
    opciones_props = [
        "Esfuerzo Máximo (MPa)", 
        "Deformación Máxima (%)", 
        "Carga Máxima (N)", 
        "Módulo de Young (GPa)", 
        "Límite de Cedencia 0.2% (MPa)"
    ]
    props_sel = []
    for prop in opciones_props:
        if st.checkbox(prop, value=True, key=f"pr_{prop}"): props_sel.append(prop)

st.divider()
st.subheader("4. Mapeo")
row_dummy = df_meta.iloc[0]
df_dummy = pd.read_csv(io.StringIO(row_dummy['__RAW_DATA__']))
headers_datos = [c.strip().replace('"', '') for c in df_dummy.columns]

idx_stress = next((i for i, h in enumerate(headers_datos) if "Esfuerzo" in h or "Stress" in h), 0)
idx_strain = next((i for i, h in enumerate(headers_datos) if "Deformación" in h or "Strain" in h), 0)

c_m1, c_m2, c_m3 = st.columns(3)
col_y_name = c_m1.selectbox("Columna Esfuerzo (Y):", headers_datos, index=idx_stress)
col_x_name = c_m2.selectbox("Columna Deformación (X):", headers_datos, index=idx_strain)
es_porcentaje = c_m3.checkbox("¿Deformación en %?", value=True)

# ==========================================
# 4. CÁLCULO
# ==========================================
st.divider()
st.subheader("🔍 Inspección y Ajuste Individual")

if probetas_a_procesar:
    col_sel_prob, col_status = st.columns([1, 2])
    with col_sel_prob:
        probeta_visual = st.selectbox("Seleccionar Probeta:", probetas_a_procesar)
    
    esta_calculada = probeta_visual in st.session_state['resultados_cache']
    
    if esta_calculada:
        st.success(f"✅ {probeta_visual}: Calculada.")
        if st.button("🔄 Recalcular"):
            del st.session_state['resultados_cache'][probeta_visual]
            st.rerun()
            
        datos_corr = st.session_state['resultados_cache'][probeta_visual]
        X_plot = datos_corr['X_final']
        Y_plot = datos_corr['Y_final']
        slope_f = datos_corr['slope']
        
        y_max_real = np.max(Y_plot)
        y_techo = y_max_real * 1.1
        # Asegurar que x_limit sea coherente con la pendiente
        x_limit = y_techo / slope_f if slope_f != 0 else np.max(X_plot)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_plot, y=Y_plot, mode='lines', name='Curva Corregida', line=dict(color='#2ca02c', width=3)))
        
        # Referencias
        x_line = np.linspace(0, x_limit, 50)
        y_young = slope_f * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_young, mode='lines', name=f'Young {slope_f/1000:.1f} GPa', line=dict(color='orange', dash='dash')))
        
        offset = 0.002
        y_offset = slope_f * (x_line - offset)
        mask_off = (y_offset >= 0) & (y_offset <= y_techo)
        fig.add_trace(go.Scatter(x=x_line[mask_off], y=y_offset[mask_off], mode='lines', name='Cedencia 0.2%', line=dict(color='red', dash='dot')))
        
        fig.update_layout(title=f"Resultado: {probeta_visual}", xaxis_title="Deformación (mm/mm)", yaxis_title="Esfuerzo (MPa)", template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Modo Ajuste
        row_viz = df_meta[df_meta['ID_Muestra'] == probeta_visual].iloc[0]
        df_viz = pd.read_csv(io.StringIO(row_viz['__RAW_DATA__']))
        df_viz.columns = [c.strip().replace('"', '') for c in df_viz.columns]
        
        Y_raw = pd.to_numeric(df_viz[col_y_name], errors='coerce').fillna(0).values
        X_raw = pd.to_numeric(df_viz[col_x_name], errors='coerce').fillna(0).values
        if es_porcentaje: X_raw = X_raw / 100.0
        
        auto_range = sugerir_rango_elastico(X_raw, Y_raw)
        
        st.info("⚠️ Ajuste pendiente.")
        modo_ajuste = st.radio("Método:", ["✅ Automático", "🛠️ Manual"], horizontal=True)
        
        rango_final = auto_range
        fig = go.Figure()
        
        if modo_ajuste == "✅ Automático":
            fig.add_trace(go.Scatter(x=X_raw, y=Y_raw, mode='lines', name='Datos Crudos', line=dict(color='gray')))
            mask_auto = (X_raw >= auto_range[0]) & (X_raw <= auto_range[1])
            fig.add_trace(go.Scatter(x=X_raw[mask_auto], y=Y_raw[mask_auto], mode='lines', name='Auto', line=dict(color='#2ca02c', width=4)))
        else:
            max_x = float(X_raw.max()) if len(X_raw) > 0 else 0.1
            rango_usuario = st.slider("Zona Elástica:", 0.0, max_x/3, value=auto_range, step=0.0001, format="%.4f")
            rango_final = rango_usuario
            
            fig.add_trace(go.Scatter(x=X_raw, y=Y_raw, mode='lines', name='Datos', line=dict(color='lightgray')))
            mask_man = (X_raw >= rango_usuario[0]) & (X_raw <= rango_usuario[1])
            fig.add_trace(go.Scatter(x=X_raw[mask_man], y=Y_raw[mask_man], mode='lines', name='Manual', line=dict(color='blue', width=4)))
            
            zoom_max = rango_usuario[1] * 2.5
            fig.update_layout(xaxis_range=[0, zoom_max], yaxis_range=[0, np.max(Y_raw)*0.8])

        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Calcular y Guardar", type="primary"):
            res_calc, X_f, Y_f, m_f = calcular_propiedades_finales(X_raw, Y_raw, rango_final, props_sel)
            
            if res_calc:
                # Carga Máxima Raw
                if "Carga Máxima (N)" in props_sel:
                    df_viz = pd.read_csv(io.StringIO(row_viz['__RAW_DATA__']))
                    df_viz.columns = [c.strip().replace('"', '') for c in df_viz.columns]
                    col_load = next((c for c in df_viz.columns if "Carga" in c or "Load" in c), None)
                    res_calc["Carga Máxima (N)"] = df_viz[col_load].max() if col_load else 0

                res_calc["ID_Muestra"] = probeta_visual
                
                # Ajuste unidades % si no se hizo antes
                if "Deformación Máxima (%)" in res_calc and not es_porcentaje:
                     res_calc["Deformación Máxima (%)"] = res_calc["Deformación Máxima (%)"] 
                
                # Factores Geométricos para reconstrucción
                max_stress = np.max(Y_raw) if np.max(Y_raw) > 0 else 1
                df_temp = pd.read_csv(io.StringIO(row_viz['__RAW_DATA__']))
                df_temp.columns = [c.strip().replace('"', '') for c in df_temp.columns]
                c_load = next((c for c in df_temp.columns if "Carga" in c or "Load" in c), None)
                c_ext = next((c for c in df_temp.columns if "Exten" in c or "Displa" in c), None)
                
                factor_area = 1.0
                factor_len = 1.0
                if c_load: factor_area = df_temp[c_load].max() / max_stress
                # Para longitud usamos el ratio aproximado en el máximo
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
                st.error("Rango inválido. Intenta seleccionar más puntos.")

# ==========================================
# 5. RESULTADOS Y EXCEL (CON CORRECCIÓN DE ERROR)
# ==========================================
st.divider()
st.subheader("📊 Tabla de Resultados Acumulados")

ids_calculados = list(st.session_state['resultados_cache'].keys())

if ids_calculados:
    lista_res = [st.session_state['resultados_cache'][k]['resultados'] for k in ids_calculados]
    df_resultados = pd.DataFrame(lista_res)
    
    cols_geo_validas = ['ID_Muestra'] + [c for c in meta_sel if c in df_meta.columns]
    df_geo_final = df_meta[df_meta['ID_Muestra'].isin(ids_calculados)][cols_geo_validas]
    
    df_final_show = pd.merge(df_geo_final, df_resultados, on="ID_Muestra")
    st.dataframe(df_final_show, use_container_width=True)
    
    total_objetivo = len(probetas_a_procesar)
    calculadas_validas = [pid for pid in ids_calculados if pid in probetas_a_procesar]
    total_listas = len(calculadas_validas)
    
    progreso = total_listas / total_objetivo if total_objetivo > 0 else 0
    st.write(f"Progreso: {total_listas} de {total_objetivo} probetas.")
    st.progress(progreso)
    
    if total_listas == total_objetivo and total_objetivo > 0:
        st.success("¡Análisis completado! Generando Excel...")
        
        # --- GENERACIÓN DE EXCEL ROBUSTA ---
        buffer_excel = io.BytesIO()
        
        # Verificar si existe xlsxwriter
        try:
            import xlsxwriter
            engine_str = 'xlsxwriter'
            has_xlsxwriter = True
        except ImportError:
            engine_str = 'openpyxl' # Fallback estándar
            has_xlsxwriter = False
            st.warning("⚠️ Librería 'xlsxwriter' no detectada. Se generará el Excel SIN gráficos. Para gráficos, instala: `pip install xlsxwriter`")

        with pd.ExcelWriter(buffer_excel, engine=engine_str) as writer:
            # 1. HOJA RESUMEN
            df_final_show.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Gráficos solo si xlsxwriter está disponible
            if has_xlsxwriter:
                workbook = writer.book
                ws_resumen = writer.sheets['Resumen']
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
                chart.set_title({'name': 'Comparativa de Curvas Corregidas'})
                chart.set_x_axis({'name': 'Deformación (mm/mm)'})
                chart.set_y_axis({'name': 'Esfuerzo (MPa)'})
            
            # 2. HOJAS INDIVIDUALES
            colores = ['#FF0000', '#0000FF', '#008000', '#FFA500', '#800080', '#00CED1', '#FF1493', '#8B4513']
            
            for i, pid in enumerate(calculadas_validas):
                cache = st.session_state['resultados_cache'][pid]
                X_f = cache['X_final']
                Y_f = cache['Y_final']
                f_area = cache['factor_area']
                f_len = cache['factor_len']
                
                # Reconstruir Carga y Extensión (Escalado simple desde la curva corregida)
                Load_corr = Y_f * f_area
                Ext_corr = X_f * f_len
                
                df_ind = pd.DataFrame({
                    "Carga (N)": Load_corr,
                    "Extensión (mm)": Ext_corr,
                    "Esfuerzo (MPa)": Y_f,
                    "Deformación (mm/mm)": X_f
                })
                
                safe_name = pid.replace("Probeta ", "P_").replace(":", "")[:30]
                df_ind.to_excel(writer, sheet_name=safe_name, index=False)
                
                if has_xlsxwriter:
                    n_rows = len(df_ind)
                    color_code = colores[i % len(colores)]
                    # Col D=Deformación (indice 3), Col C=Esfuerzo (indice 2)
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
            label="📥 Descargar Reporte Excel (.xlsx)",
            data=buffer_excel.getvalue(),
            file_name="Reporte_Instron_Completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    elif total_objetivo > 0:
        st.warning("Completa el cálculo para descargar.")

else:
    st.info("Aún no hay resultados.")