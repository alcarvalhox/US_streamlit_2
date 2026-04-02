import subprocess
import sys
import os
import base64

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
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.stApp { background-color: #003865; }
h1, h2, h3, p, label, .stMarkdown, .stText { color: white !important; }
.stTabs [data-baseweb="tab-list"] { gap: 20px; }
.stTabs [data-baseweb="tab"] { background-color: transparent !important; color: #FFC600 !important; }
.stTabs [aria-selected="true"] { color: white !important; border-bottom-color: #FFC600 !important; }
.stButton>button, [data-testid="stDownloadButton"] button {
    background-color: #FFC600; color: #003865 !important; font-weight: bold; border: none; border-radius: 5px; margin-top: 5px;
}
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover { background-color: #e6b300; color: white !important; }
.stDataFrame { background-color: white; border-radius: 5px; }
</style>
"""
st.markdown(cores_mrs, unsafe_allow_html=True)

def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

col_logo, col_titulo, col_veiculo = st.columns([1, 4, 1])

with col_logo:
    try:
        img_logo = load_image_b64("logo.png")
        st.markdown(f'<div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 20px;"><img src="data:image/png;base64,{img_logo}" style="width: 160px; height: 110px; object-fit: contain; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);"></div>', unsafe_allow_html=True)
    except:
        st.warning("⚠️ Logo não encontrada.")
        
with col_titulo:
    st.markdown("<h1 style='text-align: center; margin-top: -10px;'>Detecção Automática de Defeitos de Inspeções de US</h1>", unsafe_allow_html=True)
    
with col_veiculo:
    try:
        img_veiculo = load_image_b64("veiculo_us.jpg")
        st.markdown(f'<div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 20px;"><img src="data:image/jpeg;base64,{img_veiculo}" style="width: 160px; height: 110px; object-fit: cover; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);"></div>', unsafe_allow_html=True)
    except:
        st.warning("⚠️ Imagem do veículo não encontrada.")

st.markdown("<hr style='margin-top: -5px; margin-bottom: 15px;'>", unsafe_allow_html=True)

MODEL_DIR = "modelo"
MODEL_PT = os.path.join(MODEL_DIR, "best_alma_1.pt")
MODEL_OV = os.path.join(MODEL_DIR, "best_alma_1_openvino_model")

if 'deteccoes' not in st.session_state: st.session_state.deteccoes = []
if 'img_gallery' not in st.session_state: st.session_state.img_gallery = []
if 'page' not in st.session_state: st.session_state.page = 0
if 'audit_idx' not in st.session_state: st.session_state.audit_idx = 0 
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0 

# =====================================================================
# 2. ROTINAS DE PARIDADE TOTAL E OTIMIZAÇÃO (CACHE E OPENCV)
# =====================================================================
@st.cache_data
def ler_e_preparar_dados(arquivos):
    for f in arquivos: f.seek(0)
    df_raw = pd.concat([pd.read_csv(f) for f in arquivos]).sort_values(by='odo')
    df_raw['odo'] = (df_raw['odo'] * 1000000).astype(int)
    df_esq = df_raw[df_raw['probe'].isin([0, 6, 8, 4, 10]) & (df_raw['level'] > 450)]
    df_dir = df_raw[df_raw['probe'].isin([1, 7, 9, 5, 11]) & (df_raw['level'] > 450)]
    return df_esq, df_dir

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
    width, height = 1500, 500
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    probe_to_bgr = { 0: (0, 255, 255), 1: (0, 255, 255), 6: (0, 128, 0), 7: (0, 128, 0), 8: (128, 0, 128), 9: (128, 0, 128), 4: (0, 0, 255), 5: (0, 0, 255), 10: (255, 0, 0), 11: (255, 0, 0) }
    odos = df_win['odo'].values
    depths = df_win['depth'].values
    probes = df_win['probe'].values
    x_coords = ((odos - start) / 2400.0 * width).astype(int)
    y_coords = ((depths - 53) / 126.0 * height).astype(int)
    size = 6
    base_triangle = np.array([[0, -size], [-size, size], [size, size]], dtype=np.int32)
    for x, y, p in zip(x_coords, y_coords, probes):
        if 0 <= x < width and 0 <= y < height:
            cv2.fillPoly(img, [base_triangle + [x, y]], probe_to_bgr.get(p, (0, 255, 255)))
    return img

# =====================================================================
# 3. INTERFACE, CONTROLES E VALIDAÇÃO DE ARQUIVOS
# =====================================================================
col_upload, col_botoes = st.columns([3, 1])

with col_upload:
    files = st.file_uploader(
        "Faça o Upload dos arquivos CSV", 
        type=['csv'], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}" 
    )
    
    arquivos_prontos = False
    if files:
        colunas_esperadas = {'odo', 'frame', 'probe', 'depth', 'sample', 'level'}
        arquivos_validos = True
        
        for f in files:
            try:
                df_header = pd.read_csv(f, nrows=0)
                colunas_atuais = set(df_header.columns)
                
                if not colunas_esperadas.issubset(colunas_atuais):
                    st.error(f"⚠️ **Erro no arquivo:** `{f.name}`\n\nO formato está fora do padrão aceito. O arquivo deve conter obrigatoriamente as colunas: `odo`, `frame`, `probe`, `depth`, `sample`, `level`.")
                    arquivos_validos = False
                    break 
            except Exception as e:
                st.error(f"⚠️ **Erro de Leitura:** Não foi possível ler o arquivo `{f.name}`. Verifique se ele não está corrompido.")
                arquivos_validos = False
                break
                
        if arquivos_validos:
            st.success(f"✅ {len(files)} arquivo(s) validado(s) e dentro do padrão!")
            arquivos_prontos = True

with col_botoes:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_run = st.button("🚀 Iniciar Inferência", type="primary", use_container_width=True, disabled=not arquivos_prontos)
    
    if st.button("🧹 Limpar Arquivos", use_container_width=True):
        st.session_state.uploader_key += 1 
        st.rerun()

    if st.button("🗑️ Resetar Sistema", use_container_width=True):
        st.session_state.deteccoes = []
        st.session_state.img_gallery = []
        st.session_state.page = 0 
        st.session_state.audit_idx = 0 
        st.session_state.uploader_key += 1 
        st.cache_resource.clear()
        st.cache_data.clear() 
        st.rerun()

# =====================================================================
# 4. PROCESSAMENTO E INFERÊNCIA COM DESENHO CUSTOMIZADO
# =====================================================================
if btn_run and arquivos_prontos:
    st.session_state.page = 0 
    st.session_state.audit_idx = 0 
    model = load_ov_model()
    if model:
        progress_bar = st.progress(0.0, text="Lendo e preparando arquivos CSV da memória cache...")
        df_esq_raw, df_dir_raw = ler_e_preparar_dados(files)
        
        progress_bar.progress(0.05, text="Removendo ruídos e pontos isolados...")
        df_esq = remover_pontos_isolados(df_esq_raw)
        df_dir = remover_pontos_isolados(df_dir_raw)
        
        found = []
        gallery = []
        lados = [("Trilho_Esq", df_esq), ("Trilho_Dir", df_dir)]
        total_steps = sum([len(range(int(df['odo'].min()), int(df['odo'].max()), 2400)) for _, df in lados if not df.empty])
        
        passo_atual = 0

        for lado_nome, df_side in lados:
            if df_side.empty: continue
            steps = range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400)
            for start in steps:
                passo_atual += 1
                progress_bar.progress(min(0.05 + (0.95 * passo_atual / max(1, total_steps)), 1.0), text=f"Analisando {lado_nome}: ODO {start}mm...")

                end = start + 2400
                df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] <= end)]
                
                if len(df_win) > 5:
                    img = generate_bscan_buffer(df_win, start, end)
                    
                    # --- NMS AJUSTADO PARA EVITAR DUPLICIDADE ---
                    # iou=0.4: Suprime caixas com mais de 40% de sobreposição
                    # agnostic_nms=True: Suprime caixas sobrepostas independente da classe
                    results = model.predict(img, verbose=False, conf=0.3, iou=0.4, agnostic_nms=True)
                    
                    if len(results[0].boxes) > 0:
                        img_draw = img.copy()
                        h, w, _ = img.shape
                        local_id = 1
                        
                        for box in results[0].boxes:
                            bx = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = bx.astype(int)
                            classe_nome = model.names[int(box.cls)]
                            
                            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            texto = f"#{local_id} {classe_nome}"
                            (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(img_draw, (x1, y1 - 25), (x1 + w_txt + 5, y1), (0, 0, 255), -1)
                            cv2.putText(img_draw, texto, (x1 + 2, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            px1, px2 = x1/w, x2/w
                            py1, py2 = y1/h, y2/h
                            center_x_mm = (start + int(2400 * px1) + start + int(2400 * px2)) / 2
                            center_y_mm = (53 + int(126 * py1) + 53 + int(126 * py2)) / 2
                            comprimento = int(np.sqrt((int(2400 * px2) - int(2400 * px1))**2 + (int(126 * py2) - int(126 * py1))**2))
                            
                            found.append({
                                'ID_Global': len(found),
                                'ID_Img': f"#{local_id}",
                                'Lado': lado_nome,
                                'Classe': classe_nome,
                                'ODO_Ref': start,
                                'Coordenada ODO(mm)': int(center_x_mm),
                                'Coordenada Depth(mm)': int(center_y_mm),
                                'Comprimento(mm)': comprimento,
                                'Confiança': f"{float(box.conf):.2%}",
                                'Aprovado': True
                            })
                            local_id += 1
                            
                        gallery.append({
                            "img": img_draw, 
                            "label": f"{lado_nome} @ {start}", 
                            "odo_ref": start, 
                            "lado": lado_nome
                        })
        
        progress_bar.progress(1.0, text=f"✅ Inferência concluída! {len(found)} defeitos encontrados.")
        st.session_state.deteccoes = found
        st.session_state.img_gallery = gallery
        st.rerun() 

# =====================================================================
# 5. DISPLAY DE RESULTADOS (3 ABAS: TABELA, AUDITORIA, GALERIA)
# =====================================================================
if st.session_state.deteccoes:
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    
    df_raw = pd.DataFrame(st.session_state.deteccoes)
    
    if 'Aprovado' not in df_raw.columns:
        df_raw['Aprovado'] = True
    if 'ID_Global' not in df_raw.columns:
        df_raw['ID_Global'] = df_raw.index
    if 'ID_Img' not in df_raw.columns:
        df_raw['ID_Img'] = "#-"
    
    df_aprovados = df_raw[df_raw['Aprovado'] == True].drop(columns=['ID_Global', 'Aprovado', 'ID_Img'])
    
    aba_dados, aba_auditoria, aba_galeria = st.tabs(["📊 Tabelas e Filtros", "✅ Auditoria de Falsos Positivos", "🖼️ Galeria Geral"])
    
    # -----------------------------------------------------------------
    # ABA 1: TABELAS
    # -----------------------------------------------------------------
    with aba_dados:
        col_resumo, col_filtros = st.columns([1, 2])
        
        with col_resumo:
            st.markdown("##### 📈 Resumo Geral (Aprovados)")
            if not df_aprovados.empty:
                contagem_classes = df_aprovados['Classe'].value_counts().reset_index()
                contagem_classes.columns = ['Tipo de Defeito', 'Quantidade']
                st.dataframe(contagem_classes, hide_index=True, use_container_width=True)
            else:
                st.info("Nenhum defeito aprovado.")
            
        with col_filtros:
            st.markdown("##### 🔍 Refinar Busca")
            cf1, cf2 = st.columns(2)
            with cf1:
                classes_disponiveis = df_aprovados['Classe'].unique() if not df_aprovados.empty else []
                filtro_classe = st.multiselect("Filtrar por Classe:", options=classes_disponiveis, default=classes_disponiveis)
            with cf2:
                lados_disponiveis = df_aprovados['Lado'].unique() if not df_aprovados.empty else []
                filtro_lado = st.multiselect("Filtrar por Lado:", options=lados_disponiveis, default=lados_disponiveis)
                
        if not df_aprovados.empty:
            df_filtrado = df_aprovados[(df_aprovados['Classe'].isin(filtro_classe)) & (df_aprovados['Lado'].isin(filtro_lado))]
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_down, col_vazia2 = st.columns([1, 3])
            with col_down:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_filtrado.to_excel(writer, index=False, sheet_name='Defeitos_US')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Baixar Dados Validados", 
                    data=excel_data, 
                    file_name=f"relatorio_us_{datetime.now().strftime('%d%m%H%M')}.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            st.dataframe(df_filtrado, hide_index=True, use_container_width=True)
            
    # -----------------------------------------------------------------
    # ABA 2: AUDITORIA (NAVEGAÇÃO POR SETAS)
    # -----------------------------------------------------------------
    with aba_auditoria:
        st.markdown("##### Selecione a imagem para avaliar os defeitos encontrados:")
        
        total_imagens_det = len(st.session_state.img_gallery)
        if total_imagens_det > 0:
            
            col_nav_esq, col_nav_centro, col_nav_dir = st.columns([1, 2, 1])
            with col_nav_esq:
                if st.session_state.audit_idx > 0:
                    if st.button("⬅️ Imagem Anterior", use_container_width=True, key="btn_audit_prev"):
                        st.session_state.audit_idx -= 1
                        st.rerun()
            with col_nav_centro:
                st.markdown(f"<h5 style='text-align: center; color: white; margin-top: 10px;'>Imagem {st.session_state.audit_idx + 1} de {total_imagens_det}</h5>", unsafe_allow_html=True)
            with col_nav_dir:
                if st.session_state.audit_idx < total_imagens_det - 1:
                    if st.button("Próxima Imagem ➡️", use_container_width=True, key="btn_audit_next"):
                        st.session_state.audit_idx += 1
                        st.rerun()
            
            if st.session_state.audit_idx >= total_imagens_det:
                st.session_state.audit_idx = 0
                
            img_idx = st.session_state.audit_idx
            img_atual = st.session_state.img_gallery[img_idx]
            
            st.markdown("<br>", unsafe_allow_html=True) 
            
            col_esq, col_dir = st.columns([3, 2])
            
            with col_esq:
                st.image(img_atual['img'], channels="BGR", use_container_width=True)
                st.caption(f"Visualizando: {img_atual['label']}")
                
            with col_dir:
                st.markdown("#### Validar Detecções")
                st.markdown("Desmarque a caixa para excluir um Falso Positivo do Relatório Final.")
                
                if 'odo_ref' in img_atual and 'lado' in img_atual:
                    mask = (df_raw['ODO_Ref'] == img_atual['odo_ref']) & (df_raw['Lado'] == img_atual['lado'])
                    df_imagem_atual = df_raw[mask].copy()
                    
                    # --- TABELA DE AUDITORIA ATUALIZADA (INCLUINDO CONFIANÇA) ---
                    edited_df = st.data_editor(
                        df_imagem_atual[['ID_Global', 'ID_Img', 'Classe', 'Confiança', 'Comprimento(mm)', 'Aprovado']],
                        column_config={
                            "Aprovado": st.column_config.CheckboxColumn("✅ Aprovado?", default=True),
                            "ID_Global": None, 
                            "ID_Img": st.column_config.TextColumn("Ref na Imagem")
                        },
                        disabled=['ID_Img', 'Classe', 'Confiança', 'Comprimento(mm)'], # Trava edição das colunas informativas
                        hide_index=True,
                        use_container_width=True,
                        key=f"editor_img_{img_idx}" 
                    )
                    
                    for _, row in edited_df.iterrows():
                        g_id = int(row['ID_Global'])
                        if st.session_state.deteccoes[g_id].get('Aprovado', True) != row['Aprovado']:
                            st.session_state.deteccoes[g_id]['Aprovado'] = row['Aprovado']
                            st.rerun() 
                else:
                    st.warning("Detectamos dados antigos de galeria nesta sessão. Clique em 'Resetar Sistema' e faça a inferência novamente para usar a Auditoria.")
        else:
            st.info("Nenhuma detecção para auditar.")

    # -----------------------------------------------------------------
    # ABA 3: GALERIA DE IMAGENS
    # -----------------------------------------------------------------
    with aba_galeria:
        st.markdown("##### Dica: Passe o mouse sobre a imagem e clique no ícone de expansão no canto superior direito para visualizá-la em tela cheia.")
        
        itens_por_pagina = 20
        total_imagens = len(st.session_state.img_gallery)
        total_paginas = max(1, (total_imagens - 1) // itens_por_pagina + 1)
        
        inicio_idx = st.session_state.page * itens_por_pagina
        fim_idx = inicio_idx + itens_por_pagina
        imagens_atuais = st.session_state.img_gallery[inicio_idx:fim_idx]
        
        cols = st.columns(5)
        for idx, item in enumerate(imagens_atuais):
            with cols[idx % 5]:
                st.image(item['img'], channels="BGR", use_container_width=True)
                odo_val = item['label'].split('@')[1].strip()
                st.markdown(f"<div style='text-align: center; color: #FFC600; font-weight: bold; margin-top: -10px; margin-bottom: 15px;'>ODO: {odo_val}</div>", unsafe_allow_html=True)
        
        if total_paginas > 1:
            st.write("") 
            col_pg_esq, col_pg_centro, col_pg_dir = st.columns([1, 2, 1])
            with col_pg_esq:
                if st.session_state.page > 0:
                    if st.button("⬅️ Anterior", use_container_width=True):
                        st.session_state.page -= 1
                        st.rerun()
            with col_pg_centro:
                st.markdown(f"<h5 style='text-align: center; color: white; margin-top: 10px;'>Mostrando imagens {inicio_idx + 1} a {min(fim_idx, total_imagens)} (Página {st.session_state.page + 1} de {total_paginas})</h5>", unsafe_allow_html=True)
            with col_pg_dir:
                if st.session_state.page < total_paginas - 1:
                    if st.button("Próxima ➡️", use_container_width=True):
                        st.session_state.page += 1
                        st.rerun()
