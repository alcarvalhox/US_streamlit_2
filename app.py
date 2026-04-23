import subprocess
import sys
import os
import tempfile
import base64
import warnings

# =====================================================================
# BLOQUEIO DE LOGS, TELEMETRIA E "PHONE HOME" DO YOLO
# (Deve ocorrer estritamente antes da importação da biblioteca)
# =====================================================================
# 1. Cria a pasta fisicamente na área temporária antes do YOLO tentar
TMP_DIR = tempfile.gettempdir()
YOLO_CFG_DIR = os.path.join(TMP_DIR, 'Ultralytics_Config')
os.makedirs(YOLO_CFG_DIR, exist_ok=True) 

# 2. Injeta as chaves de isolamento offline no sistema
os.environ['YOLO_CONFIG_DIR'] = YOLO_CFG_DIR
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLO_TELEMETRY'] = 'False'    # Desliga a telemetria (evita erro de conexão)
os.environ['YOLO_UPDATE_CHECK'] = 'False' # Impede checagem de atualizações
os.environ['YOLO_SYNC'] = 'False'         # Impede sincronização online

# =====================================================================
# 0. AUTO-INSTALAÇÃO DE DEPENDÊNCIAS
# =====================================================================
def install_dependencies():
    if os.path.exists("requirements.txt"):
        try:
            if 'dependencies_installed' not in os.environ:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "scipy"])
                os.environ['dependencies_installed'] = '1'
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")

if __name__ == "__main__":
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

# =====================================================================
# 1. CONFIGURAÇÕES GLOBAIS E FUNÇÕES BASE
# =====================================================================
CONFIG_MODELOS = {
    "Alma": "best_alma_2.pt",
    "Boleto": "best_boleto_1.pt",
    "Patim": "best_patim_1.pt"
}

LIMITES_PROFUNDIDADE = {
    "Boleto": (0, 52),
    "Alma": (53, 179),
    "Patim": (180, 223)
}

SONDAS_ESQUERDA = [0, 6, 8, 4, 10]
SONDAS_DIREITA = [1, 7, 9, 5, 11]
MODEL_DIR = "modelo"

def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def remover_pontos_isolados(df, raio=15):
    if df.empty: return df
    coords = df[['odo', 'depth']].values
    tree = cKDTree(coords)
    contagem = tree.query_ball_point(coords, r=raio, return_length=True)
    return df[contagem > 1].copy() 

@st.cache_resource
def load_yolo_model(nome_pt):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    path_pt = os.path.join(MODEL_DIR, nome_pt)
    
    if os.path.exists(path_pt):
        return YOLO(path_pt, task='segment', verbose=False)
    return None 

def generate_bscan_buffer(df_win, start, end, min_depth, max_depth):
    width, height = 1500, 500
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    probe_to_bgr = { 0: (0, 255, 255), 1: (0, 255, 255), 6: (0, 128, 0), 7: (0, 128, 0), 8: (128, 0, 128), 9: (128, 0, 128), 4: (0, 0, 255), 5: (0, 0, 255), 10: (255, 0, 0), 11: (255, 0, 0) }
    
    odos = df_win['odo'].values
    depths = df_win['depth'].values
    probes = df_win['probe'].values
    
    delta_depth = max_depth - min_depth
    if delta_depth <= 0: delta_depth = 1 
    
    x_coords = ((odos - start) / 2400.0 * width).astype(int)
    y_coords = ((depths - min_depth) / float(delta_depth) * height).astype(int)
    
    size_x = 10
    size_y = 5
    base_triangle = np.array([[0, -size_y], [-size_x, size_y], [size_x, size_y]], dtype=np.int32)
    
    for x, y, p in zip(x_coords, y_coords, probes):
        if 0 <= x < width and 0 <= y < height:
            cv2.fillPoly(img, [base_triangle + [x, y]], probe_to_bgr.get(p, (0, 255, 255)))
            
    return img

def gerar_zip_dataset():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in st.session_state.img_gallery:
            if 'img_clean' not in item: continue
            local = item.get('local', 'Alma') 
            nome_base = f"{item['lado']}_{item['odo_ref']}"
            
            _, buffer_img = cv2.imencode(".jpg", item['img_clean'])
            zip_file.writestr(f"{local}/images/{nome_base}.jpg", buffer_img.tobytes())
            
            linhas_yolo = []
            for det in st.session_state.deteccoes:
                if det['ODO_Ref'] == item['odo_ref'] and det['Lado'] == item['lado'] and det.get('Local', 'Alma') == local and det['Aprovado']:
                    linhas_yolo.append(det['yolo_bbox'])
            zip_file.writestr(f"{local}/labels/{nome_base}.txt", "\n".join(linhas_yolo))
            
    return zip_buffer.getvalue()

# =====================================================================
# 2. FUNÇÃO PRINCIPAL DA INTERFACE
# =====================================================================
def main():
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
    div.row-widget.stRadio > div{flex-direction:row; justify-content: center;}
    div.row-widget.stRadio > div > label{
        background-color: #FFC600; padding: 10px 30px; border-radius: 5px; color:#003865 !important; font-weight: bold; margin-right: 15px; cursor: pointer; border: 2px solid transparent;
    }
    div.row-widget.stRadio > div > label[data-checked="true"] {
        border: 2px solid white; background-color: #e6b300;
    }
    </style>
    """
    st.markdown(cores_mrs, unsafe_allow_html=True)

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

    if 'deteccoes' not in st.session_state: st.session_state.deteccoes = []
    if 'img_gallery' not in st.session_state: st.session_state.img_gallery = []
    if 'page' not in st.session_state: st.session_state.page = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
    if 'audit_idx' not in st.session_state: st.session_state.audit_idx = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0} 
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0 

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
                st.success(f"✅ {len(files)} arquivo(s) validados! Os ecos serão fisicamente separados por profundidade para análise.")
                arquivos_prontos = True

    with col_botoes:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_run = st.button("🚀 Iniciar Inferências", type="primary", use_container_width=True, disabled=not arquivos_prontos)
        
        if st.button("🧹 Limpar Caixa de Upload", use_container_width=True):
            st.session_state.uploader_key += 1 
            st.rerun()

        if st.button("🗑️ Resetar Sistema", use_container_width=True):
            st.session_state.deteccoes = []
            st.session_state.img_gallery = []
            st.session_state.page = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
            st.session_state.audit_idx = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
            st.session_state.uploader_key += 1 
            st.cache_resource.clear()
            st.cache_data.clear() 
            st.rerun()

    if btn_run and arquivos_prontos:
        st.session_state.page = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
        st.session_state.audit_idx = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
        
        modelos_ativos = {}
        for local_nome, pt_file in CONFIG_MODELOS.items():
            m = load_yolo_model(pt_file) 
            if m: modelos_ativos[local_nome] = m
                
        if not modelos_ativos:
            st.error("Nenhum modelo (.pt) encontrado na pasta 'modelo'. Adicione pelo menos um para realizar a inferência.")
            st.stop()
        
        progress_bar = st.progress(0.0, text="Concatenando arquivos e aplicando limpeza de ruídos...")
        
        for f in files: f.seek(0)
        df_raw = pd.concat([pd.read_csv(f) for f in files]).sort_values(by='odo')
        df_raw['odo'] = (df_raw['odo'] * 1000000).astype(int)
        df_raw = df_raw[df_raw['level'] > 450]
        df_raw = remover_pontos_isolados(df_raw) 
        
        found = []
        gallery = []
        
        total_steps = len(modelos_ativos) * len(range(int(df_raw['odo'].min()), int(df_raw['odo'].max()), 2400))
        passo_atual = 0

        for local_nome, model in modelos_ativos.items():
            min_depth, max_depth = LIMITES_PROFUNDIDADE.get(local_nome, (0, 223))
            delta_depth = max_depth - min_depth if (max_depth - min_depth) > 0 else 1
            
            df_local = df_raw[(df_raw['depth'] >= min_depth) & (df_raw['depth'] <= max_depth)]
            if df_local.empty: continue

            df_esq = df_local[df_local['probe'].isin(SONDAS_ESQUERDA)]
            df_dir = df_local[df_local['probe'].isin(SONDAS_DIREITA)]
            lados = [("Trilho_Esq", df_esq), ("Trilho_Dir", df_dir)]
            
            for lado_nome, df_side in lados:
                if df_side.empty: continue
                
                for start in range(int(df_side['odo'].min()), int(df_side['odo'].max()), 2400):
                    passo_atual += 1
                    if total_steps > 0:
                        progress_bar.progress(min(0.05 + (0.95 * passo_atual / total_steps), 1.0), text=f"Analisando {local_nome} ({lado_nome}): ODO {start}mm...")
                    
                    end = start + 2400
                    df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] <= end)]
                    
                    if len(df_win) > 5:
                        img_base = generate_bscan_buffer(df_win, start, end, min_depth, max_depth)
                        img_clean = img_base.copy()
                        
                        results = model.predict(img_clean, verbose=False, conf=0.05)
                        
                        if len(results[0].boxes) > 0:
                            raw_dets = []
                            h_img, w_img = img_clean.shape[:2]
                            
                            masks_xy = results[0].masks.xy if hasattr(results[0], 'masks') and results[0].masks is not None else [None] * len(results[0].boxes)
                            
                            for i, box in enumerate(results[0].boxes):
                                mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
                                poly = masks_xy[i]
                                
                                if poly is not None and len(poly) > 0:
                                    cv2.fillPoly(mask_u8, [np.array(poly, dtype=np.int32)], 1)
                                    
                                raw_dets.append({
                                    'box': box.xyxy[0].cpu().numpy().astype(int),
                                    'conf': float(box.conf),
                                    'cls_nome': model.names[int(box.cls)],
                                    'cls_id': int(box.cls),
                                    'mask': mask_u8 > 0 
                                })
                            
                            raw_dets.sort(key=lambda x: x['conf'], reverse=True)
                            
                            dets_mescladas = []
                            for d in raw_dets:
                                bx1, suprimido = d['box'], False
                                for f in dets_mescladas:
                                    bx2 = f['box']
                                    xl, yt = max(bx1[0], bx2[0]), max(bx1[1], bx2[1])
                                    xr, yb = min(bx1[2], bx2[2]), min(bx1[3], bx2[3])
                                    
                                    if xr > xl and yb > yt:
                                        area_inter = (xr-xl)*(yb-yt)
                                        area_min = min((bx1[2]-bx1[0])*(bx1[3]-bx1[1]), (bx2[2]-bx2[0])*(bx2[3]-bx2[1]))
                                        limite_nms = 0.80 if d['cls_nome'] == 'Tala_Isolada' else 0.40 
                                        
                                        if (area_inter / area_min) > limite_nms:
                                            if d['cls_id'] == f['cls_id']:
                                                f['box'][0] = min(bx1[0], bx2[0])
                                                f['box'][1] = min(bx1[1], bx2[1])
                                                f['box'][2] = max(bx1[2], bx2[2])
                                                f['box'][3] = max(bx1[3], bx2[3])
                                                f['mask'] = f['mask'] | d['mask']
                                            suprimido = True
                                            break
                                if not suprimido: 
                                    dets_mescladas.append(d)

                            # =========================================================
                            # FILTRO DE COR E DIMENSÃO PARA A CLASSE 'Furo'
                            # =========================================================
                            valid_dets = []
                            for d in dets_mescladas:
                                x1_orig, y1_orig, x2_orig, y2_orig = d['box']
                                px1, px2, py1, py2 = x1_orig/w_img, x2_orig/w_img, y1_orig/h_img, y2_orig/h_img
                                
                                largura_mm = int(abs(px2 - px1) * 2400)
                                altura_fisica_ref = {"Alma": 129, "Boleto": 52, "Patim": 43}.get(local_nome, 129)
                                altura_mm = int(abs(py2 - py1) * altura_fisica_ref)
                                
                                if local_nome == 'Alma' and d['cls_nome'] == 'Furo':
                                    # 1. Checa a proporção
                                    if largura_mm <= 130 or altura_mm <= 15:
                                        continue
                                        
                                    # 2. Checa as cores na matriz da imagem original (BGR)
                                    roi = img_clean[y1_orig:y2_orig, x1_orig:x2_orig]
                                    tem_verde = np.any(np.all(roi == [0, 128, 0], axis=-1))
                                    tem_roxo = np.any(np.all(roi == [128, 0, 128], axis=-1))
                                    
                                    if not (tem_verde and tem_roxo):
                                        continue
                                        
                                valid_dets.append((d, largura_mm, altura_mm, px1, px2, py1, py2))

                            if valid_dets:
                                VIS_W, VIS_H = 2400, 400
                                img_draw = cv2.resize(img_clean, (VIS_W, VIS_H), interpolation=cv2.INTER_LINEAR)
                                
                                for local_id, (d, largura_mm, altura_mm, px1, px2, py1, py2) in enumerate(valid_dets, 1):
                                    x1_orig, y1_orig, x2_orig, y2_orig = d['box']
                                    area_caixa = max(1, x2_orig - x1_orig) * max(1, y2_orig - y1_orig)
                                    
                                    x1 = int((x1_orig / w_img) * VIS_W)
                                    y1 = int((y1_orig / h_img) * VIS_H)
                                    x2 = int((x2_orig / w_img) * VIS_W)
                                    y2 = int((y2_orig / h_img) * VIS_H)
                                    
                                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0,0,255), 2)
                                    cv2.putText(img_draw, f"#{local_id} {d['cls_nome']}", (x1+2, max(15, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                                    
                                    center_x_mm = (start + int(2400 * px1) + start + int(2400 * px2)) / 2
                                    center_y_mm = (min_depth + int(delta_depth * py1) + min_depth + int(delta_depth * py2)) / 2
                                    
                                    found.append({
                                        'ID_Global': len(found), 
                                        'ID_Img': f"#{local_id}", 
                                        'Local': local_nome, 
                                        'Lado': lado_nome, 
                                        'Classe': d['cls_nome'], 
                                        'ODO_Ref': start,
                                        'Coordenada ODO(mm)': int(center_x_mm),
                                        'Coordenada Depth(mm)': int(center_y_mm),
                                        'Largura(mm)': largura_mm,
                                        'Altura(mm)': altura_mm,
                                        'Área (px)': int(area_caixa),
                                        'Confiança': f"{d['conf']:.2%}", 
                                        'Aprovado': True,
                                        'yolo_bbox': f"{d['cls_id']} {((x1_orig+x2_orig)/2)/w_img:.6f} {((y1_orig+y2_orig)/2)/h_img:.6f} {(x2_orig-x1_orig)/w_img:.6f} {(y2_orig-y1_orig)/h_img:.6f}"
                                    })
                                gallery.append({"img": img_draw, "img_clean": img_clean, "label": f"{lado_nome} @ {start}", "odo_ref": start, "lado": lado_nome, "local": local_nome})
        
        progress_bar.progress(1.0, text="✅ Processamento concluído!")
        
        if found:
            st.session_state.deteccoes = found
            st.session_state.img_gallery = gallery
            st.rerun()
        else:
            st.info("A inferência foi processada com sucesso, mas a rede neural não encontrou nenhum defeito nas imagens geradas.")

    if st.session_state.deteccoes or st.session_state.img_gallery:
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
        
        df_raw = pd.DataFrame(st.session_state.deteccoes) if st.session_state.deteccoes else pd.DataFrame(columns=['Local', 'Aprovado', 'ID_Global', 'ID_Img'])
        
        if not df_raw.empty and 'Local' not in df_raw.columns: df_raw['Local'] = "Alma" 
        if not df_raw.empty and 'Aprovado' not in df_raw.columns: df_raw['Aprovado'] = True
        if not df_raw.empty and 'ID_Global' not in df_raw.columns: df_raw['ID_Global'] = df_raw.index
            
        st.markdown("<h4 style='color: #FFC600; text-align: center;'>Alternar Visualização de Tabelas e Modelos:</h4>", unsafe_allow_html=True)
        
        locais_fixos = ["Alma", "Boleto", "Patim", "🌐 Visão Global"]
        local_selecionado = st.radio("Selecione:", locais_fixos, horizontal=True, label_visibility="collapsed")
        
        if local_selecionado == "🌐 Visão Global":
            aba_dados = st.tabs(["📊 Tabelas e Relatório Global"])[0]
            with aba_dados:
                col_resumo, col_filtros = st.columns([1, 2])
                df_aprovados_global = df_raw[df_raw['Aprovado'] == True].copy() if not df_raw.empty else pd.DataFrame()
                
                with col_resumo:
                    st.markdown("##### 📈 Resumo Geral (Aprovados)")
                    if not df_aprovados_global.empty:
                        contagem_classes = df_aprovados_global['Classe'].value_counts().reset_index()
                        contagem_classes.columns = ['Tipo de Defeito', 'Quantidade']
                        st.dataframe(contagem_classes, hide_index=True)
                    else:
                        st.info("Nenhum defeito aprovado na via toda.")
                
                with col_filtros:
                    st.markdown("##### 🔍 Refinar Busca Global")
                    cf1, cf2, cf3 = st.columns(3)
                    with cf1:
                        classes_disp = df_aprovados_global['Classe'].unique() if not df_aprovados_global.empty else []
                        filtro_classe = st.multiselect("Classe:", options=classes_disp, default=classes_disp)
                    with cf2:
                        lados_disp = df_aprovados_global['Lado'].unique() if not df_aprovados_global.empty else []
                        filtro_lado = st.multiselect("Lado:", options=lados_disp, default=lados_disp)
                    with cf3:
                        loc_disp = df_aprovados_global['Local'].unique() if not df_aprovados_global.empty else []
                        filtro_local = st.multiselect("Local:", options=loc_disp, default=loc_disp)
                        
                if not df_aprovados_global.empty:
                    df_filtrado_global = df_aprovados_global[
                        (df_aprovados_global['Classe'].isin(filtro_classe)) & 
                        (df_aprovados_global['Lado'].isin(filtro_lado)) &
                        (df_aprovados_global['Local'].isin(filtro_local))
                    ]
                    
                    colunas_esconder = ['ID_Global', 'Aprovado', 'ID_Img', 'yolo_bbox']
                    df_final_export = df_filtrado_global.drop(columns=colunas_esconder, errors='ignore')
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(df_final_export, hide_index=True)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    col_down, _ = st.columns([1, 2])
                    with col_down:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_final_export.to_excel(writer, index=False, sheet_name='Defeitos_US_Globais')
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 Baixar Relatório Unificado (Excel)", 
                            data=excel_data, 
                            file_name=f"relatorio_us_unificado_{datetime.now().strftime('%d%m%H%M')}.xlsx", 
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            type="primary"
                        )
        else:
            df_local_atual = df_raw[df_raw['Local'] == local_selecionado].copy() if not df_raw.empty else pd.DataFrame()
            galeria_local_atual = [item for item in st.session_state.img_gallery if item.get('local') == local_selecionado]
            
            if local_selecionado not in st.session_state.audit_idx: st.session_state.audit_idx[local_selecionado] = 0
            if local_selecionado not in st.session_state.page: st.session_state.page[local_selecionado] = 0
            
            colunas_esconder = ['ID_Global', 'Aprovado', 'ID_Img', 'yolo_bbox']
            df_aprovados_local = df_local_atual[df_local_atual['Aprovado'] == True].drop(columns=colunas_esconder, errors='ignore') if not df_local_atual.empty else pd.DataFrame()
            
            aba_dados, aba_auditoria, aba_galeria = st.tabs(["📊 Tabelas e Filtros", "✅ Auditoria & Retreinamento", "🖼️ Galeria Geral"])
            
            with aba_dados:
                col_resumo, col_filtros = st.columns([1, 2])
                with col_resumo:
                    st.markdown(f"##### 📈 Resumo - {local_selecionado} (Aprovados)")
                    if not df_aprovados_local.empty:
                        contagem_classes = df_aprovados_local['Classe'].value_counts().reset_index()
                        contagem_classes.columns = ['Tipo de Defeito', 'Quantidade']
                        st.dataframe(contagem_classes, hide_index=True)
                    else:
                        st.info(f"Nenhum defeito aprovado em {local_selecionado} no momento.")
                    
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
                    st.dataframe(df_filtrado_local, hide_index=True)

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
                        st.image(img_atual['img'], channels="BGR")
                        st.caption(f"Visualizando: {img_atual['label']}")
                        
                    with col_dir:
                        st.markdown("#### Validar Detecções")
                        mask = (df_raw['ODO_Ref'] == img_atual['odo_ref']) & (df_raw['Lado'] == img_atual['lado']) & (df_raw['Local'] == local_selecionado)
                        df_imagem_atual = df_raw[mask].copy()
                        
                        if not df_imagem_atual.empty:
                            edited_df = st.data_editor(
                                df_imagem_atual[['ID_Global', 'ID_Img', 'Classe', 'Coordenada Depth(mm)', 'Área (px)', 'Confiança', 'Largura(mm)', 'Altura(mm)', 'Aprovado']],
                                column_config={
                                    "Aprovado": st.column_config.CheckboxColumn("✅ Aprovado?", default=True),
                                    "ID_Global": None, 
                                    "ID_Img": st.column_config.TextColumn("Ref")
                                },
                                disabled=['ID_Img', 'Classe', 'Coordenada Depth(mm)', 'Área (px)', 'Confiança', 'Largura(mm)', 'Altura(mm)'], 
                                hide_index=True,
                                key=f"editor_img_{local_selecionado}_{img_idx}" 
                            )
                            
                            for _, row in edited_df.iterrows():
                                g_id = int(row['ID_Global'])
                                if st.session_state.deteccoes[g_id].get('Aprovado', True) != row['Aprovado']:
                                    st.session_state.deteccoes[g_id]['Aprovado'] = row['Aprovado']
                                    st.rerun() 
                        else:
                            st.info("Esta imagem não possui detecções ativas. Serve como negative mining no retreinamento.")
                    
                    st.markdown("<br><hr>", unsafe_allow_html=True)
                    st.markdown("#### 📦 Exportar Dataset Estruturado para Retreinamento")
                    
                    if 'img_clean' in st.session_state.img_gallery[0]:
                        st.download_button(
                            label="⬇️ Baixar Dataset Global Estruturado",
                            data=gerar_zip_dataset(),
                            file_name=f"dataset_multi_retreino_{datetime.now().strftime('%d%m%H%M')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                else:
                    if local_selecionado == "Patim":
                        st.info("⚠️ O modelo do Patim não está carregado ou os arquivos não possuem dados dessa região.")
                    else:
                        st.info(f"Nenhuma imagem correspondente a {local_selecionado} para auditar.")

            with aba_galeria:
                st.markdown("##### Dica: Passe o mouse sobre a imagem e clique no ícone de expansão para tela cheia.")
                itens_por_pagina = 20
                total_paginas = max(1, (len(galeria_local_atual) - 1) // itens_por_pagina + 1) if len(galeria_local_atual) > 0 else 1
                
                if st.session_state.page[local_selecionado] >= total_paginas:
                    st.session_state.page[local_selecionado] = 0
                    
                inicio_idx = st.session_state.page[local_selecionado] * itens_por_pagina
                fim_idx = inicio_idx + itens_por_pagina
                imagens_atuais = galeria_local_atual[inicio_idx:fim_idx]
                
                cols = st.columns(3) 
                for idx, item in enumerate(imagens_atuais):
                    with cols[idx % 3]:
                        st.image(item['img'], channels="BGR")
                        odo_val = item['label'].split('@')[1].strip()
                        st.markdown(f"<div style='text-align: center; color: #FFC600; font-weight: bold; margin-top: -10px; margin-bottom: 15px;'>ODO: {odo_val}</div>", unsafe_allow_html=True)
                
                if total_paginas > 1:
                    st.write("") 
                    col_pg_esq, col_pg_centro, col_pg_dir = st.columns([1, 2, 1])
                    with col_pg_esq:
                        if st.session_state.page[local_selecionado] > 0:
                            if st.button("⬅️ Anterior", use_container_width=True, key="pg_gal_prev"):
                                st.session_state.page[local_selecionado] -= 1
                                st.rerun()
                    with col_pg_centro:
                        st.markdown(f"<h5 style='text-align: center; color: white; margin-top: 10px;'>Página {st.session_state.page[local_selecionado] + 1} de {total_paginas}</h5>", unsafe_allow_html=True)
                    with col_pg_dir:
                        if st.session_state.page[local_selecionado] < total_paginas - 1:
                            if st.button("Próxima ➡️", use_container_width=True, key="pg_gal_next"):
                                st.session_state.page[local_selecionado] += 1
                                st.rerun()

# =====================================================================
# 3. GATILHO DE EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    main()
