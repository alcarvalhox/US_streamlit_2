import subprocess
import sys
import os
import base64

# =====================================================================
# 0. AUTO-INSTALAÇÃO DE DEPENDÊNCIAS E CONFIGURAÇÃO
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
import zipfile
from ultralytics import YOLO
from datetime import datetime
from scipy.spatial import cKDTree

CONFIG_MODELOS = {
    "Alma": "best_alma_1.pt",
    "Boleto": "best_boleto_1.pt",
    "Patim": "best_patim_1.pt"
}

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
div.row-widget.stRadio > div{flex-direction:row;}
div.row-widget.stRadio > div > label{
    background-color: #FFC600; padding: 10px 20px; border-radius: 5px; color:#003865 !important; font-weight: bold; margin-right: 10px; cursor: pointer;
}
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

if 'deteccoes' not in st.session_state: st.session_state.deteccoes = []
if 'img_gallery' not in st.session_state: st.session_state.img_gallery = []
if 'page' not in st.session_state: st.session_state.page = {"Alma": 0, "Boleto": 0, "Patim": 0}
if 'audit_idx' not in st.session_state: st.session_state.audit_idx = {"Alma": 0, "Boleto": 0, "Patim": 0} 
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
def load_ov_model(nome_pt):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    path_pt = os.path.join(MODEL_DIR, nome_pt)
    path_ov = os.path.join(MODEL_DIR, nome_pt.replace(".pt", "_openvino_model"))
    
    if not os.path.exists(path_ov):
        if os.path.exists(path_pt):
            with st.status(f"Convertendo pesos do {nome_pt} para OpenVINO local...") as s:
                model = YOLO(path_pt)
                model.export(format="openvino", half=True)
                s.update(label="Otimização Concluída!", state="complete")
        else:
            return None 
    return YOLO(path_ov, task='detect')

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

def gerar_zip_dataset():
    """Gera um arquivo ZIP exportando a estrutura de pastas reais para cada local."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in st.session_state.img_gallery:
            if 'img_clean' not in item: continue
            
            local = item.get('local', 'Alma') # Proteção para dados antigos
            nome_base = f"{item['lado']}_{item['odo_ref']}"
            
            # Salva na pasta do respectivo modelo/região (ex: Alma/images/Trilho_Esq_100.jpg)
            _, buffer_img = cv2.imencode(".jpg", item['img_clean'])
            zip_file.writestr(f"{local}/images/{nome_base}.jpg", buffer_img.tobytes())
            
            linhas_yolo = []
            for det in st.session_state.deteccoes:
                if det['ODO_Ref'] == item['odo_ref'] and det['Lado'] == item['lado'] and det.get('Local', 'Alma') == local and det['Aprovado']:
                    linhas_yolo.append(det['yolo_bbox'])
            
            zip_file.writestr(f"{local}/labels/{nome_base}.txt", "\n".join(linhas_yolo))
            
    return zip_buffer.getvalue()

# =====================================================================
# 3. INTERFACE DE UPLOAD E VALIDAÇÃO GERAL
# =====================================================================
col_upload, col_botoes = st.columns([3, 1])

with col_upload:
    files = st.file_uploader(
        "Faça o Upload dos arquivos CSV de inspeção brutos", 
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
                if not colunas_esperadas.issubset(set(df_header.columns)):
                    st.error(f"⚠️ **Erro no arquivo:** `{f.name}` - Formato fora do padrão.")
                    arquivos_validos = False
                    break 
            except:
                st.error(f"⚠️ **Erro de Leitura:** Não foi possível ler o arquivo `{f.name}`.")
                arquivos_validos = False
                break
                
        if arquivos_validos:
            st.success(f"✅ {len(files)} arquivo(s) validado(s) e prontos para geração das imagens e análise!")
            arquivos_prontos = True

with col_botoes:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_run = st.button("🚀 Iniciar Inferências", type="primary", use_container_width=True, disabled=not arquivos_prontos)
    
    if st.button("🧹 Limpar Arquivos", use_container_width=True):
        st.session_state.uploader_key += 1 
        st.rerun()

    if st.button("🗑️ Resetar Sistema", use_container_width=True):
        st.session_state.deteccoes = []
        st.session_state.img_gallery = []
        st.session_state.page = {"Alma": 0, "Boleto": 0, "Patim": 0}
        st.session_state.audit_idx = {"Alma": 0, "Boleto": 0, "Patim": 0}
        st.session_state.uploader_key += 1 
        st.cache_resource.clear()
        st.cache_data.clear() 
        st.rerun()

# =====================================================================
# 4. PROCESSAMENTO EM LOTE (ENGINE MULTI-MODELO PARALELO)
# =====================================================================
if btn_run and arquivos_prontos:
    st.session_state.page = {"Alma": 0, "Boleto": 0, "Patim": 0}
    st.session_state.audit_idx = {"Alma": 0, "Boleto": 0, "Patim": 0}
    
    # 4.1 Carrega os modelos ativos na pasta 'modelo'
    modelos_ativos = {}
    for local_nome, pt_file in CONFIG_MODELOS.items():
        m = load_ov_model(pt_file)
        if m: 
            modelos_ativos[local_nome] = m
            
    if not modelos_ativos:
        st.error("Nenhum modelo (.pt) encontrado na pasta 'modelo'. Adicione pelo menos um para realizar a inferência.")
        st.stop()
    
    progress_bar = st.progress(0.0, text="Lendo e gerando representação B-scan dos arquivos CSV...")
    
    df_esq_raw, df_dir_raw = ler_e_preparar_dados(files)
    df_esq = remover_pontos_isolados(df_esq_raw)
    df_dir = remover_pontos_isolados(df_dir_raw)
    
    found = []
    gallery = []
    lados = [("Trilho_Esq", df_esq), ("Trilho_Dir", df_dir)]
    total_steps = sum([len(range(int(df['odo'].min()), int(df['odo'].max()), 2400)) for _, df in lados if not df.empty])
    
    passo_atual = 0
    for lado_nome, df_side in lados:
        if df_side.empty: continue
        for start in range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400):
            passo_atual += 1
            progress_bar.progress(min(0.05 + (0.95 * passo_atual / max(1, total_steps)), 1.0), text=f"Aplicando Inteligência ({lado_nome}): ODO {start}mm...")
            
            end = start + 2400
            df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] <= end)]
            
            if len(df_win) > 5:
                # Gera a imagem base UMA ÚNICA VEZ
                img_base = generate_bscan_buffer(df_win, start, end)
                
                # Passa a mesma imagem para cada modelo ativo isoladamente
                for local_nome, model in modelos_ativos.items():
                    img_clean = img_base.copy()
                    results = model.predict(img_clean, verbose=False, conf=0.5)
                    
                    if len(results[0].boxes) > 0:
                        raw_dets = []
                        for box in results[0].boxes:
                            raw_dets.append({'box': box.xyxy[0].cpu().numpy().astype(int), 'conf': float(box.conf), 'cls_nome': model.names[int(box.cls)], 'cls_id': int(box.cls)})
                        
                        raw_dets.sort(key=lambda x: x['conf'], reverse=True)
                        final_dets = []
                        for d in raw_dets:
                            bx1, duplicado = d['box'], False
                            for f in final_dets:
                                bx2 = f['box']
                                xl, yt, xr, yb = max(bx1[0], bx2[0]), max(bx1[1], bx2[1]), min(bx1[2], bx2[2]), min(bx1[3], bx2[3])
                                if xr > xl and yb > yt:
                                    if ((xr-xl)*(yb-yt) / min((bx1[2]-bx1[0])*(bx1[3]-bx1[1]), (bx2[2]-bx2[0])*(bx2[3]-bx2[1]))) > 0.10:
                                        duplicado = True; break
                            if not duplicado: final_dets.append(d)

                        if final_dets:
                            img_draw = img_clean.copy()
                            h, w, _ = img_draw.shape
                            for local_id, d in enumerate(final_dets, 1):
                                x1, y1, x2, y2 = d['box']
                                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0,0,255), 2)
                                cv2.putText(img_draw, f"#{local_id} {d['cls_nome']}", (x1+2, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                                
                                px1, px2, py1, py2 = x1/w, x2/w, y1/h, y2/h
                                found.append({
                                    'ID_Global': len(found), 
                                    'ID_Img': f"#{local_id}", 
                                    'Local': local_nome, 
                                    'Lado': lado_nome, 
                                    'Classe': d['cls_nome'], 
                                    'ODO_Ref': start,
                                    'Coordenada ODO(mm)': int((start+int(2400*px1)+start+int(2400*px2))/2),
                                    'Coordenada Depth(mm)': int((53+int(126*py1)+53+int(126*py2))/2),
                                    'Comprimento(mm)': int(np.sqrt((int(2400*px2)-int(2400*px1))**2 + (int(126*py2)-int(126*py1))**2)),
                                    'Confiança': f"{d['conf']:.2%}", 
                                    'Aprovado': True,
                                    'yolo_bbox': f"{d['cls_id']} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                                })
                            gallery.append({"img": img_draw, "img_clean": img_clean, "label": f"{lado_nome} @ {start}", "odo_ref": start, "lado": lado_nome, "local": local_nome})
    
    progress_bar.progress(1.0, text=f"✅ Inferência concluída pelos modelos {', '.join(modelos_ativos.keys())}!")
    
    if found:
        st.session_state.deteccoes = found
        st.session_state.img_gallery = gallery
        st.rerun()
    else:
        st.info("A inferência foi processada, mas a rede neural não encontrou defeitos em nenhum modelo ativo.")

# =====================================================================
# 5. DISPLAY DE RESULTADOS E NAVEGAÇÃO
# =====================================================================
if st.session_state.deteccoes:
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    
    df_raw = pd.DataFrame(st.session_state.deteccoes)
    
    # --- PROTEÇÃO CONTRA CACHE ANTIGO ---
    if 'Local' not in df_raw.columns:
        df_raw['Local'] = "Alma" # Atribui Alma para dados velhos da memória
    if 'Aprovado' not in df_raw.columns:
        df_raw['Aprovado'] = True
    if 'ID_Global' not in df_raw.columns:
        df_raw['ID_Global'] = df_raw.index
    if 'ID_Img' not in df_raw.columns:
        df_raw['ID_Img'] = "#-"
        
    locais_disponiveis = list(df_raw['Local'].unique())
    
    st.markdown("<h4 style='color: #FFC600;'>Visualizar Resultados do Modelo:</h4>", unsafe_allow_html=True)
    local_selecionado = st.radio("Selecione:", locais_disponiveis, horizontal=True, label_visibility="collapsed")
    
    df_local_atual = df_raw[df_raw['Local'] == local_selecionado].copy()
    galeria_local_atual = [item for item in st.session_state.img_gallery if item.get('local', 'Alma') == local_selecionado]
    
    if local_selecionado not in st.session_state.audit_idx: st.session_state.audit_idx[local_selecionado] = 0
    if local_selecionado not in st.session_state.page: st.session_state.page[local_selecionado] = 0
    
    colunas_esconder = ['ID_Global', 'Aprovado', 'ID_Img', 'yolo_bbox']
    df_aprovados_local = df_local_atual[df_local_atual['Aprovado'] == True].drop(columns=colunas_esconder, errors='ignore')
    
    aba_dados, aba_auditoria, aba_galeria = st.tabs(["📊 Tabelas e Filtros", "✅ Auditoria & Retreinamento", "🖼️ Galeria Geral"])
    
    # -----------------------------------------------------------------
    # ABA 1: TABELAS
    # -----------------------------------------------------------------
    with aba_dados:
        col_resumo, col_filtros = st.columns([1, 2])
        
        with col_resumo:
            st.markdown(f"##### 📈 Resumo - {local_selecionado} (Aprovados)")
            if not df_aprovados_local.empty:
                contagem_classes = df_aprovados_local['Classe'].value_counts().reset_index()
                contagem_classes.columns = ['Tipo de Defeito', 'Quantidade']
                st.dataframe(contagem_classes, hide_index=True, use_container_width=True)
            else:
                st.info("Nenhum defeito aprovado neste local.")
            
        with col_filtros:
            st.markdown("##### 🔍 Refinar Busca")
            cf1, cf2 = st.columns(2)
            with cf1:
                classes_disp = df_aprovados_local['Classe'].unique() if not df_aprovados_local.empty else []
                filtro_classe = st.multiselect("Filtrar por Classe:", options=classes_disp, default=classes_disp)
            with cf2:
                lados_disp = df_aprovados_local['Lado'].unique() if not df_aprovados_local.empty else []
                filtro_lado = st.multiselect("Filtrar por Lado:", options=lados_disp, default=lados_disp)
                
        if not df_aprovados_local.empty:
            df_filtrado_local = df_aprovados_local[(df_aprovados_local['Classe'].isin(filtro_classe)) & (df_aprovados_local['Lado'].isin(filtro_lado))]
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_down, col_vazia2 = st.columns([1, 3])
            
            with col_down:
                df_aprovados_global = df_raw[df_raw['Aprovado'] == True].drop(columns=['ID_Global', 'Aprovado', 'ID_Img', 'yolo_bbox'], errors='ignore')
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_aprovados_global.to_excel(writer, index=False, sheet_name='Defeitos_US')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Baixar Relatório Completo (Todos os Modelos)", 
                    data=excel_data, 
                    file_name=f"relatorio_us_completo_{datetime.now().strftime('%d%m%H%M')}.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            st.dataframe(df_filtrado_local, hide_index=True, use_container_width=True)
            
    # -----------------------------------------------------------------
    # ABA 2: AUDITORIA E EXPORTAÇÃO
    # -----------------------------------------------------------------
    with aba_auditoria:
        total_imagens_det = len(galeria_local_atual)
        if total_imagens_det > 0:
            
            col_nav_esq, col_nav_centro, col_nav_dir = st.columns([1, 2, 1])
            with col_nav_esq:
                if st.session_state.audit_idx[local_selecionado] > 0:
                    if st.button("⬅️ Imagem Anterior", use_container_width=True, key="btn_prev"):
                        st.session_state.audit_idx[local_selecionado] -= 1
                        st.rerun()
            with col_nav_centro:
                st.markdown(f"<h5 style='text-align: center; color: white; margin-top: 10px;'>Imagem {st.session_state.audit_idx[local_selecionado] + 1} de {total_imagens_det} ({local_selecionado})</h5>", unsafe_allow_html=True)
            with col_nav_dir:
                if st.session_state.audit_idx[local_selecionado] < total_imagens_det - 1:
                    if st.button("Próxima Imagem ➡️", use_container_width=True, key="btn_next"):
                        st.session_state.audit_idx[local_selecionado] += 1
                        st.rerun()
            
            img_idx = st.session_state.audit_idx[local_selecionado]
            img_atual = galeria_local_atual[img_idx]
            
            st.markdown("<br>", unsafe_allow_html=True) 
            col_esq, col_dir = st.columns([3, 2])
            
            with col_esq:
                st.image(img_atual['img'], channels="BGR", use_container_width=True)
                st.caption(f"Visualizando: {img_atual['label']}")
                
            with col_dir:
                st.markdown("#### Validar Detecções")
                
                mask = (df_raw['ODO_Ref'] == img_atual['odo_ref']) & (df_raw['Lado'] == img_atual['lado']) & (df_raw['Local'] == local_selecionado)
                df_imagem_atual = df_raw[mask].copy()
                
                edited_df = st.data_editor(
                    df_imagem_atual[['ID_Global', 'ID_Img', 'Classe', 'Coordenada Depth(mm)', 'Confiança', 'Comprimento(mm)', 'Aprovado']],
                    column_config={
                        "Aprovado": st.column_config.CheckboxColumn("✅ Aprovado?", default=True),
                        "ID_Global": None, 
                        "ID_Img": st.column_config.TextColumn("Ref na Imagem")
                    },
                    disabled=['ID_Img', 'Classe', 'Coordenada Depth(mm)', 'Confiança', 'Comprimento(mm)'], 
                    hide_index=True,
                    use_container_width=True,
                    key=f"editor_img_{local_selecionado}_{img_idx}" 
                )
                
                for _, row in edited_df.iterrows():
                    g_id = int(row['ID_Global'])
                    if st.session_state.deteccoes[g_id].get('Aprovado', True) != row['Aprovado']:
                        st.session_state.deteccoes[g_id]['Aprovado'] = row['Aprovado']
                        st.rerun() 
            
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.markdown("#### 📦 Exportar Dataset Estruturado para Retreinamento (YOLOv8)")
            st.markdown("Baixe um único ZIP contendo as imagens limpas e anotações corrigidas, automaticamente separadas nas pastas físicas de cada local (Alma, Boleto, Patim).")
            
            if 'img_clean' in st.session_state.img_gallery[0]:
                st.download_button(
                    label="⬇️ Baixar Dataset Global Estruturado",
                    data=gerar_zip_dataset(),
                    file_name=f"dataset_multi_retreino_{datetime.now().strftime('%d%m%H%M')}.zip",
                    mime="application/zip",
                    use_container_width=False
                )
        else:
            st.info("Nenhuma detecção para auditar nesta visualização.")

    # -----------------------------------------------------------------
    # ABA 3: GALERIA DE IMAGENS
    # -----------------------------------------------------------------
    with aba_galeria:
        st.markdown("##### Dica: Passe o mouse sobre a imagem e clique no ícone de expansão para tela cheia.")
        
        itens_por_pagina = 20
        total_paginas = max(1, (total_imagens_det - 1) // itens_por_pagina + 1)
        
        if st.session_state.page[local_selecionado] >= total_paginas:
            st.session_state.page[local_selecionado] = 0
            
        inicio_idx = st.session_state.page[local_selecionado] * itens_por_pagina
        fim_idx = inicio_idx + itens_por_pagina
        imagens_atuais = galeria_local_atual[inicio_idx:fim_idx]
        
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
                if st.session_state.page[local_selecionado] > 0:
                    if st.button("⬅️ Anterior", use_container_width=True, key="pg_prev"):
                        st.session_state.page[local_selecionado] -= 1
                        st.rerun()
            with col_pg_centro:
                st.markdown(f"<h5 style='text-align: center; color: white; margin-top: 10px;'>Página {st.session_state.page[local_selecionado] + 1} de {total_paginas}</h5>", unsafe_allow_html=True)
            with col_pg_dir:
                if st.session_state.page[local_selecionado] < total_paginas - 1:
                    if st.button("Próxima ➡️", use_container_width=True, key="pg_next"):
                        st.session_state.page[local_selecionado] += 1
                        st.rerun()
