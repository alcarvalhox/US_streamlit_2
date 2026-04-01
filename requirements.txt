import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import io
import shutil
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

# =====================================================================
# 1. CONFIGURAÇÕES E ESTILO
# =====================================================================
st.set_page_config(page_title="Sistema de Inspeção US - Local", layout="wide")

MODEL_DIR = "models"
MODEL_PT = os.path.join(MODEL_DIR, "best_alma_1.pt")
MODEL_OV = os.path.join(MODEL_DIR, "best_alma_1_openvino_model")
LIMITES_DEPTH = {'alma': (53, 179)}

# Inicialização do estado da sessão
if 'deteccoes' not in st.session_state:
    st.session_state.deteccoes = []
if 'img_gallery' not in st.session_state:
    st.session_state.img_gallery = []

# =====================================================================
# 2. ROTINAS ORIGINAIS (PARIDADE TOTAL)
# =====================================================================
def remover_pontos_isolados(df, raio=10):
    """Rotina 5.5 idêntica ao original."""
    if df.empty: return df
    coords = df[['odo', 'depth']].values
    tree = cKDTree(coords)
    contagem = tree.query_ball_point(coords, r=raio, return_length=True)
    return df[contagem > 1].copy()

@st.cache_resource
def load_ov_model():
    """Conversão automática para OpenVINO."""
    if not os.path.exists(MODEL_OV):
        if os.path.exists(MODEL_PT):
            model = YOLO(MODEL_PT)
            model.export(format="openvino", half=True)
        else:
            st.error("Modelo .pt não encontrado em /models")
            return None
    return YOLO(MODEL_OV, task='detect')

def generate_bscan_buffer(df_win, start, end):
    """Gera o vetor de imagem idêntico ao padrão de treinamento."""
    probe_colors = {0: 'yellow', 1: 'yellow', 6: 'green', 8: 'purple', 
                    4: 'red', 10: 'blue', 7: 'green', 9: 'purple', 5: 'red', 11: 'blue'}
    
    # Mantendo figsize e DPI originais para não distorcer a detecção
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    sns.scatterplot(data=df_win, x="odo", y="depth", hue="probe", 
                    palette=probe_colors, ax=ax, marker='^', s=60, legend=False)
    ax.set_xlim(start, end)
    ax.set_ylim(179, 53)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return img

# =====================================================================
# 3. INTERFACE E BOTÕES
# =====================================================================
st.title("🚇 Painel de Inspeção Ultrassônica - Coordenadoria")

# Colunas de ação superiores
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded_files = st.file_uploader("1. Carregar Arquivos CSV", accept_multiple_files=True)

with col2:
    if st.button("🗑️ Nova Inspeção (Limpar)", use_container_width=True):
        st.session_state.deteccoes = []
        st.session_state.img_gallery = []
        st.cache_data.clear()
        st.success("Dados resetados.")

with col3:
    # Botão de Execução
    run_inference = st.button("🚀 Executar Inferência", type="primary", use_container_width=True)

# =====================================================================
# 4. LÓGICA DE PROCESSAMENTO
# =====================================================================
if run_inference and uploaded_files:
    model = load_ov_model()
    if model:
        all_data = pd.concat([pd.read_csv(f) for f in uploaded_files]).sort_values(by='odo')
        all_data['odo'] = (all_data['odo'] * 1000000).astype(int)
        
        # Filtros de trilho
        df_esq = all_data[all_data['probe'].isin([0, 6, 8, 4, 10]) & (all_data['level'] > 450)]
        df_dir = all_data[all_data['probe'].isin([1, 7, 9, 5, 11]) & (all_data['level'] > 450)]
        
        # Aplicação cKDTree
        df_esq = remover_pontos_isolados(df_esq)
        df_dir = remover_pontos_isolados(df_dir)
        
        progress = st.progress(0)
        detections_found = []
        gallery = []
        
        lados = [("Esq", df_esq), ("Dir", df_dir)]
        
        for lado_nome, df_side in lados:
            if df_side.empty: continue
            odo_steps = range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400)
            
            for i, start in enumerate(odo_steps):
                end = start + 2400
                df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] <= end)]
                
                if len(df_win) > 5:
                    img = generate_bscan_buffer(df_win, start, end)
                    results = model.predict(img, verbose=False, conf=0.3)
                    
                    if len(results[0].boxes) > 0:
                        # Gerar imagem com BBox para o thumbnail
                        res_plotted = results[0].plot()
                        if len(gallery) < 20: # Limite solicitado
                            gallery.append({"img": res_plotted, "label": f"{lado_nome} ODO:{start}"})
                        
                        # Conversão de coordenadas (Pixel para MM) - Rotina Inferencia Original
                        h, w, _ = img.shape
                        for box in results[0].boxes:
                            bx = box.xyxy[0].cpu().numpy()
                            perc_x = ((bx[0] + bx[2]) / 2) / w
                            perc_y = ((bx[1] + bx[3]) / 2) / h
                            
                            detections_found.append({
                                'Trilho': lado_nome,
                                'ODO_Ref': start,
                                'ODO_Exato(mm)': int(start + (2400 * perc_x)),
                                'Profundidade(mm)': int(53 + (126 * perc_y)),
                                'Classe': model.names[int(box.cls)],
                                'Confiança': float(box.conf)
                            })
                progress.progress((i + 1) / len(odo_steps))
        
        st.session_state.deteccoes = detections_found
        st.session_state.img_gallery = gallery

# =====================================================================
# 5. EXIBIÇÃO DE RESULTADOS (THUMBNAILS E DOWNLOAD)
# =====================================================================
if st.session_state.deteccoes:
    st.markdown("---")
    df_final = pd.DataFrame(st.session_state.deteccoes)
    
    # Botão de Download do Relatório
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Relatório Final (CSV)", csv, "relatorio_inspecao.csv", "text/csv")
    
    st.subheader("🖼️ Galeria de Detecções (Top 20)")
    # Thumbnail Grid
    cols = st.columns(5)
    for idx, item in enumerate(st.session_state.img_gallery):
        with cols[idx % 5]:
            # Botão invisível ou imagem expansível
            with st.expander(f"Ver: {item['label']}"):
                st.image(item['img'], channels="BGR", use_container_width=True)
            st.caption(item['label'])

    st.subheader("📋 Dados Analíticos")
    st.dataframe(df_final, use_container_width=True)
