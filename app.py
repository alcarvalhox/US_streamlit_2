import subprocess
import sys
import os

# =====================================================================
# 0. AUTO-INSTALAÇÃO DE DEPENDÊNCIAS
# =====================================================================
def install_dependencies():
    """Verifica se o requirements.txt existe e instala as dependências."""
    if os.path.exists("requirements.txt"):
        try:
            # Verifica se já rodamos a instalação nesta sessão para evitar loops
            if 'dependencies_installed' not in os.environ:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                os.environ['dependencies_installed'] = '1'
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")

# Executa a instalação antes de importar as bibliotecas pesadas
install_dependencies()

# Agora importamos o restante
import streamlit as st
import pandas as pd
import numpy as np
import cv2
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
st.set_page_config(page_title="Sistema de Inspeção US - Coordenadoria", layout="wide")

MODEL_DIR = "models"
MODEL_PT = os.path.join(MODEL_DIR, "best_alma_1.pt")
MODEL_OV = os.path.join(MODEL_DIR, "best_alma_1_openvino_model")
LIMITES_DEPTH = {'alma': (53, 179)}

if 'deteccoes' not in st.session_state:
    st.session_state.deteccoes = []
if 'img_gallery' not in st.session_state:
    st.session_state.img_gallery = []

# =====================================================================
# 2. ROTINAS DE PARIDADE TOTAL (FILTRAGEM E CONVERSÃO)
# =====================================================================
def remover_pontos_isolados(df, raio=10):
    """Etapa 5.5 original: Filtro espacial cKDTree."""
    if df.empty: return df
    coords = df[['odo', 'depth']].values
    tree = cKDTree(coords)
    contagem = tree.query_ball_point(coords, r=raio, return_length=True)
    return df[contagem > 1].copy()

@st.cache_resource
def load_ov_model():
    """Garante que o modelo esteja em formato OpenVINO para máxima economia."""
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    if not os.path.exists(MODEL_OV):
        if os.path.exists(MODEL_PT):
            with st.status("Convertendo pesos para OpenVINO local...") as s:
                model = YOLO(MODEL_PT)
                model.export(format="openvino", half=True)
                s.update(label="Otimização Concluída!", state="complete")
        else:
            st.error(f"Erro: Coloque o arquivo {MODEL_PT} na pasta /models")
            return None
    return YOLO(MODEL_OV, task='detect')

def generate_bscan_buffer(df_win, start, end):
    """Reconstrução exata do padrão visual de treinamento (15x5 pol, 100 DPI)."""
    probe_colors = {0: 'yellow', 1: 'yellow', 6: 'green', 8: 'purple', 
                    4: 'red', 10: 'blue', 7: 'green', 9: 'purple', 5: 'red', 11: 'blue'}
    
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
# 3. INTERFACE E CONTROLES
# =====================================================================
st.title("🔍 Painel de Controle de Manutenção - Visão Computacional")

col_upload, col_reset, col_run = st.columns([2, 1, 1])

with col_upload:
    files = st.file_uploader("Upload de CSVs", accept_multiple_files=True)

with col_reset:
    if st.button("🗑️ Resetar Sistema", use_container_width=True):
        st.session_state.deteccoes = []
        st.session_state.img_gallery = []
        st.cache_resource.clear()
        st.rerun()

with col_run:
    btn_run = st.button("🚀 Iniciar Inferência", type="primary", use_container_width=True)

# =====================================================================
# 4. PROCESSAMENTO E INFERÊNCIA
# =====================================================================
if btn_run and files:
    model = load_ov_model()
    if model:
        # Ingestão e Preparação (Etapas 2, 3 e 4 originais)
        df_raw = pd.concat([pd.read_csv(f) for f in files]).sort_values(by='odo')
        df_raw['odo'] = (df_raw['odo'] * 1000000).astype(int)
        
        # Separação e Filtro Nível > 450 (Rotina Original)
        df_esq = df_raw[df_raw['probe'].isin([0, 6, 8, 4, 10]) & (df_raw['level'] > 450)]
        df_dir = df_raw[df_raw['probe'].isin([1, 7, 9, 5, 11]) & (df_raw['level'] > 450)]
        
        # Filtragem cKDTree (Etapa 5.5 original)
        df_esq = remover_pontos_isolados(df_esq)
        df_dir = remover_pontos_isolados(df_dir)
        
        found = []
        gallery = []
        progress = st.progress(0)
        
        lados = [("Trilho_Esq", df_esq), ("Trilho_Dir", df_dir)]
        
        for lado_nome, df_side in lados:
            if df_side.empty: continue
            
            steps = range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400)
            for i, start in enumerate(steps):
                end = start + 2400
                df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] <= end)]
                
                if len(df_win) > 5:
                    img = generate_bscan_buffer(df_win, start, end)
                    results = model.predict(img, verbose=False, conf=0.3)
                    
                    if len(results[0].boxes) > 0:
                        # Salva para o Thumbnail (Máx 20)
                        if len(gallery) < 20:
                            gallery.append({"img": results[0].plot(), "label": f"{lado_nome} @ {start}"})
                        
                        # Conversão Pixel -> MM (Rotina Inferencia Original)
                        h, w, _ = img.shape
                        for box in results[0].boxes:
                            bx = box.xyxy[0].cpu().numpy()
                            px, py = ((bx[0]+bx[2])/2)/w, ((bx[1]+bx[3])/2)/h
                            
                            found.append({
                                'Lado': lado_nome,
                                'ODO_Ref': start,
                                'ODO_Real(mm)': int(start + (2400 * px)),
                                'Prof_Real(mm)': int(53 + (126 * py)),
                                'Classe': model.names[int(box.cls)],
                                'Confiança': f"{float(box.conf):.2%}"
                            })
                progress.progress((i + 1) / len(steps))
        
        st.session_state.deteccoes = found
        st.session_state.img_gallery = gallery

# =====================================================================
# 5. DISPLAY DE RESULTADOS
# =====================================================================
if st.session_state.deteccoes:
    st.divider()
    df_rep = pd.DataFrame(st.session_state.deteccoes)
    
    # Download do Relatório Final
    csv = df_rep.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Relatório (CSV)", csv, f"relatorio_{datetime.now().strftime('%d%m%H%M')}.csv", "text/csv")
    
    # Galeria de Thumbnails com expansão
    st.subheader("🖼️ Detecções em Destaque (Thumbnail - Clique para ampliar)")
    cols = st.columns(5)
    for idx, item in enumerate(st.session_state.img_gallery):
        with cols[idx % 5]:
            with st.expander(f"ODO: {item['label'].split('@')[1]}"):
                st.image(item['img'], channels="BGR", use_container_width=True)
            st.caption(item['label'])

    st.subheader("📋 Tabela Analítica")
    st.dataframe(df_rep, use_container_width=True)
