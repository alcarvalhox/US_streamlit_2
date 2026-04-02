import subprocess
import sys
import os

# =====================================================================
# 0. AUTO-INSTALAÇÃO DE DEPENDÊNCIAS
# =====================================================================
def install_dependencies():
    if os.path.exists("requirements.txt"):
        try:
            if 'dependencies_installed' not in os.environ:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
                os.environ['dependencies_installed'] = '1'
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")

install_dependencies()

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

# =====================================================================
# 1. CONFIGURAÇÕES, TÍTULO E ESTILO
# =====================================================================
st.set_page_config(
    page_title="Detecção automática de defeitos de inspeções de US", 
    page_icon="logo.png", 
    layout="wide"
)

cores_mrs = """
<style>
/* Cor de fundo principal Azul Escuro */
.stApp {
    background-color: #003865;
}
/* Textos em branco para contraste */
h1, h2, h3, p, label, .stMarkdown, .stText {
    color: white !important;
}
/* Estilo dos Botões Normais e Botão de Download */
.stButton>button, [data-testid="stDownloadButton"] button {
    background-color: #FFC600;
    color: #003865 !important;
    font-weight: bold;
    border: none;
    border-radius: 5px;
}
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover {
    background-color: #e6b300; 
    color: white !important;
}
.stDataFrame {
    background-color: white;
    border-radius: 5px;
}
.streamlit-expanderHeader {
    color: #FFC600 !important;
    font-weight: bold;
}
</style>
"""
st.markdown(cores_mrs, unsafe_allow_html=True)

col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    try:
        st.image("logo.png", width=150)
    except:
        st.warning("⚠️ Arquivo 'logo.png' não encontrado na pasta principal.")
with col_titulo:
    st.title("Detecção automática de defeitos de inspeções de US")

MODEL_DIR = "modelo"
MODEL_PT = os.path.join(MODEL_DIR, "best_alma_1.pt")
MODEL_OV = os.path.join(MODEL_DIR, "best_alma_1_openvino_model")
LIMITES_DEPTH = {'alma': (53, 179)}

# Inicialização de Variáveis na Memória (Session State)
if 'deteccoes' not in st.session_state:
    st.session_state.deteccoes = []
if 'img_gallery' not in st.session_state:
    st.session_state.img_gallery = []
if 'page' not in st.session_state:
    st.session_state.page = 0

# =====================================================================
# 2. ROTINAS DE PARIDADE TOTAL (FILTRAGEM E CONVERSÃO)
# =====================================================================
def remover_pontos_isolados(df, raio=10):
    if df.empty: return df
    coords = df[['odo', 'depth']].values
    tree = cKDTree(coords)
    contagem = tree.query_ball_point(coords, r=raio, return_length=True)
    return df[contagem > 1].copy()

@st.cache_resource
def load_ov_model():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(MODEL_OV):
        if os.path.exists(MODEL_PT):
            with st.status("Convertendo pesos para OpenVINO local...") as s:
                model = YOLO(MODEL_PT)
                model.export(format="openvino", half=True)
                s.update(label="Otimização Concluída!", state="complete")
        else:
            st.error(f"Erro: Coloque o arquivo {MODEL_PT} na pasta /{MODEL_DIR}")
            return None
    return YOLO(MODEL_OV, task='detect')

def generate_bscan_buffer(df_win, start, end):
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
st.markdown("### Seleção de Dados")

# Estrutura ajustada: 2 colunas. Upload na esquerda, Botões empilhados na direita.
col_upload, col_botoes = st.columns([3, 1])

with col_upload:
    files = st.file_uploader("Faça o Upload dos arquivos CSV", accept_multiple_files=True)

with col_botoes:
    st.write("") 
    # Iniciar em cima, Resetar embaixo
    btn_run = st.button("🚀 Iniciar Inferência", type="primary", use_container_width=True)
    if st.button("🗑️ Resetar Sistema", use_container_width=True):
        st.session_state.deteccoes = []
        st.session_state.img_gallery = []
        st.session_state.page = 0 # Reseta a página da galeria também
        st.cache_resource.clear()
        st.rerun()

# =====================================================================
# 4. PROCESSAMENTO E INFERÊNCIA
# =====================================================================
if btn_run and files:
    st.session_state.page = 0 # Reseta a página sempre que iniciar nova inferência
    model = load_ov_model()
    if model:
        progress_bar = st.progress(0.0, text="Iniciando a leitura dos arquivos CSV...")
        
        df_raw = pd.concat([pd.read_csv(f) for f in files]).sort_values(by='odo')
        df_raw['odo'] = (df_raw['odo'] * 1000000).astype(int)
        
        df_esq = df_raw[df_raw['probe'].isin([0, 6, 8, 4, 10]) & (df_raw['level'] > 450)]
        df_dir = df_raw[df_raw['probe'].isin([1, 7, 9, 5, 11]) & (df_raw['level'] > 450)]
        
        df_esq = remover_pontos_isolados(df_esq)
        df_dir = remover_pontos_isolados(df_dir)
        
        found = []
        gallery = []
        lados = [("Trilho_Esq", df_esq), ("Trilho_Dir", df_dir)]
        
        total_steps = 0
        for _, df_side in lados:
            if not df_side.empty:
                total_steps += len(range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400))
        
        passo_atual = 0

        for lado_nome, df_side in lados:
            if df_side.empty: continue
            
            steps = range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400)
            for start in steps:
                passo_atual += 1
                perc = min(passo_atual / total_steps, 1.0)
                progress_bar.progress(perc, text=f"Analisando {lado_nome}: ODO {start}mm (Passo {passo_atual}/{total_steps})...")

                end = start + 2400
                df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] <= end)]
                
                if len(df_win) > 5:
                    img = generate_bscan_buffer(df_win, start, end)
                    results = model.predict(img, verbose=False, conf=0.3)
                    
                    if len(results[0].boxes) > 0:
                        # Removido o limite de 50 imagens para permitir a paginação funcionar livremente
                        gallery.append({"img": results[0].plot(), "label": f"{lado_nome} @ {start}"})
                        
                        h, w, _ = img.shape
                        for box in results[0].boxes:
                            bx = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = bx
                            px1, px2 = x1/w, x2/w
                            py1, py2 = y1/h, y2/h
                            
                            x1_t = start + int(2400 * px1)
                            x2_t = start + int(2400 * px2)
                            y1_t = 53 + int(126 * py1)
                            y2_t = 53 + int(126 * py2)
                            
                            center_x_mm = (x1_t + x2_t) / 2
                            center_y_mm = (y1_t + y2_t) / 2
                            comprimento = int(np.sqrt((x2_t - x1_t)**2 + (y2_t - y1_t)**2))
                            
                            found.append({
                                'Lado': lado_nome,
                                'Classe': model.names[int(box.cls)],
                                'ODO_Ref': start,
                                'Coordenada ODO(mm)': int(center_x_mm),
                                'Coordenada Depth(mm)': int(center_y_mm),
                                'Comprimento(mm)': comprimento,
                                'Confiança': f"{float(box.conf):.2%}"
                            })
        
        progress_bar.progress(1.0, text=f"✅ Inferência concluída! {len(found)} defeitos encontrados.")
        st.session_state.deteccoes = found
        st.session_state.img_gallery = gallery

# =====================================================================
# 5. DISPLAY DE RESULTADOS E PAGINAÇÃO
# =====================================================================
if st.session_state.deteccoes:
    st.divider()
    df_rep = pd.DataFrame(st.session_state.deteccoes)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_rep.to_excel(writer, index=False, sheet_name='Defeitos_US')
    excel_data = output.getvalue()
    
    # Botão agora obedecerá o CSS
    st.download_button(
        label="📥 Baixar Relatório (Excel)", 
        data=excel_data, 
        file_name=f"relatorio_us_{datetime.now().strftime('%d%m%H%M')}.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # --- SISTEMA DE PAGINAÇÃO DA GALERIA ---
    st.subheader("🖼️ Detecções (Galeria de Imagens)")
    st.markdown("*Dica: Clique na aba abaixo de cada Thumbnail para ver a imagem em tamanho real.*")
    
    itens_por_pagina = 20
    total_imagens = len(st.session_state.img_gallery)
    total_paginas = max(1, (total_imagens - 1) // itens_por_pagina + 1)
    
    # Define o "fatiamento" da lista de imagens
    inicio_idx = st.session_state.page * itens_por_pagina
    fim_idx = inicio_idx + itens_por_pagina
    imagens_atuais = st.session_state.img_gallery[inicio_idx:fim_idx]
    
    # Exibe as imagens da página atual
    cols = st.columns(5)
    for idx, item in enumerate(imagens_atuais):
        with cols[idx % 5]:
            st.image(item['img'], channels="BGR", use_container_width=True)
            with st.expander(f"🔎 Ampliar ODO: {item['label'].split('@')[1]}"):
                st.image(item['img'], channels="BGR", use_container_width=True)
                st.caption(item['label'])
    
    # Renderiza os botões de controle de página abaixo da galeria
    if total_paginas > 1:
        st.write("") # Espaçamento
        col_esq, col_centro, col_dir = st.columns([1, 2, 1])
        
        with col_esq:
            if st.session_state.page > 0:
                if st.button("⬅️ Página Anterior", use_container_width=True):
                    st.session_state.page -= 1
                    st.rerun()
                    
        with col_centro:
            st.markdown(f"<h5 style='text-align: center; color: white;'>Mostrando imagens {inicio_idx + 1} a {min(fim_idx, total_imagens)} (Página {st.session_state.page + 1} de {total_paginas})</h5>", unsafe_allow_html=True)
            
        with col_dir:
            if st.session_state.page < total_paginas - 1:
                if st.button("Próxima Página ➡️", use_container_width=True):
                    st.session_state.page += 1
                    st.rerun()
            
    st.divider()
    st.subheader("📋 Tabela Analítica")
    st.dataframe(df_rep, use_container_width=True)
