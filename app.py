import subprocess
import sys
import os

# =====================================================================
# 0. INSTALAÇÃO DE DEPENDÊNCIAS (SE NECESSÁRIO)
# =====================================================================
# Esta função verifica e instala as bibliotecas necessárias se você
# estiver rodando este código em um ambiente novo (como o Google Colab).
# Em um ambiente já configurado, você pode ignorar ou comentar esta parte.
def install_dependencies():
    dependencies = ["opencv-python", "numpy", "ultralytics", "openvino"]
    try:
        import cv2, numpy, ultralytics, openvino
    except ImportError:
        print("Instalando dependências necessárias...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies])
            print("Instalação concluída com sucesso. Reinicie o kernel ou re-execute o script.")
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")
            print("Por favor, instale as seguintes bibliotecas manualmente:")
            for dep in dependencies:
                print(f" - {dep}")
        #sys.exit() # Para a execução para que o usuário possa reiniciar o ambiente se necessário.

# install_dependencies() # Descomente para rodar a instalação automática.

# =====================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS E CARREGAMENTO DO MODELO
# =====================================================================
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURAÇÕES DO MODELO ---
# Defina o caminho para o seu arquivo de pesos (.pt) treinado para segmentação.
# Para este exemplo, usarei um modelo 'n' (nano) para rapidez, mas você deve
# usar o seu modelo 's' (small) otimizado em OpenVINO para máxima performance.
MODEL_WEIGHTS_PATH = "best.pt" # Substitua pelo caminho do seu modelo.

# Carregamento do modelo YOLOv8 de segmentação otimizado em OpenVINO.
try:
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {MODEL_WEIGHTS_PATH}")
    model = YOLO(MODEL_WEIGHTS_PATH, task='segment')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Por favor, treine um modelo YOLOv8 de segmentação e defina o caminho correto.")
    sys.exit()

# =====================================================================
# 2. DEFINIÇÃO DA LÓGICA DE PÓS-PROCESSAMENTO APERFEIÇOADA
# =====================================================================
# Esta função processa os resultados crus da inferência, aplicando os filtros
# geométricos e de confiança solicitados, incluindo a Solidez Geometrica.
def process_ultrasound_detections(results):
    processed_detections = []
    
    # --- AJUSTE FINO DE LIMIARES ---
    # Reduzimos levemente a confiança para recuperar omissões (prob 2 e 3).
    # O filtro de solidez cuidará de ruídos fracos.
    CONFIDENCE_THRESHOLD = 0.20 
    
    # Área total mínima do polígono em pixels. Mantido baixo para aceitar furos pequenos.
    MIN_POLYGON_AREA = 50 
    
    # ⚠️ NOVO FILTRO: SOLIDEZ GEOMETRICA
    # A área da máscara real deve ser pelo menos 30% da área do bounding box.
    # Isso remove ruídos fragmentados ou esguios (prob 1).
    MIN_SOLIDITY = 0.30 
    
    # Razão de aspecto mínima e máxima (largura / altura).
    MIN_ASPECT_RATIO = 0.2
    MAX_ASPECT_RATIO = 5.0
    
    class_names = results.names

    for i, box in enumerate(results.boxes):
        confidence = float(box.conf)
        class_id = int(box.cls)
        class_name = class_names[class_id]

        # 1. Filtro de Confiança Inicial
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Extrair dados da segmentação (polígono)
        if results.masks is not None:
            # Pegar as coordenadas do polígono (normalizadas para 0.0-1.0)
            polygon_norm = results.masks.xyn[i]
            
            # Converter para coordenadas de pixel reais
            h, w, _ = results.orig_shape
            polygon_pixel = (polygon_norm * [w, h]).astype(np.int32)
            
            # Calcular a área do polígono real em pixels.
            polygon_area = cv2.contourArea(polygon_pixel)

            # 2. Filtro de Área Mínima do Polígono (ajudará furos pequenos)
            if polygon_area < MIN_POLYGON_AREA:
                continue

            # Calcular métricas geométricas adicionais para o Bounding Box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_area = bbox_w * bbox_h
            aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0

            # ⚠️ NOVO CÁLCULO: SOLIDEZ GEOMETRICA
            # Razão entre a área real do polígono e a área do bounding box.
            # Um furo real é robusto e 'preenche' bem a sua caixa. Ruídos esguios não.
            solidity = polygon_area / bbox_area if bbox_area > 0 else 0

            # --- APLICAÇÃO DOS FILTROS FINAIS ---
            
            # 3. Filtro de Razão de Aspecto
            aspect_ratio_pass = MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
            
            # 4. ⚠️ APLICAÇÃO DO FILTRO DE SOLIDEZ (Resolve o Prob 1 de ruídos pequenos)
            solidity_pass = solidity >= MIN_SOLIDITY
            
            if aspect_ratio_pass and solidity_pass:
                processed_detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "polygon": polygon_pixel,
                    "area": polygon_area,
                    "solidity": solidity # Adicionamos para visualização
                })

    return processed_detections

# =====================================================================
# 3. VISUALIZAÇÃO DOS RESULTADOS (PREVIEW)
# =====================================================================
# Esta função desenha as detecções processadas na imagem para verificação.
def visualize_results(image_path, detections, output_filename="results_preview.png"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar imagem para visualização: {image_path}")
        return

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        solidity = det['solidity']
        
        # Cor para "Furo" e outras classes (BGR).
        color = (0, 255, 255) if class_name == "Furo" else (255, 0, 0)
        
        # Desenhar o Polígono da segmentação.
        cv2.polylines(img, [det['polygon']], isClosed=True, color=color, thickness=2)
        
        # Desenhar o Bounding Box para referência.
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)
        
        # Texto da etiqueta: Classe + Confiança + Solidez Geometrica.
        label = f"{class_name} C:{confidence:.2f} S:{solidity:.2f}"
        
        # Fundo do texto para legibilidade.
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + t_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_filename, img)
    print(f"Visualização salva como: {output_filename}")
    return img

# =====================================================================
# 4. EXECUÇÃO DO FLUXO COMPLETO NAS IMAGENS DE EXEMPLO
# =====================================================================
image_paths = [f"image_{i}.png" for i in range(10)] # Substitua pelos caminhos reais das suas imagens.

# Para este exemplo de preview, usaremos uma imagem que combine os casos
# para demonstração clara.
example_images = ["image_0.png", "image_1.png", "image_2.png"] # Substitua pelas suas imagens reais.

for i, image_path in enumerate(example_images):
    print(f"\n--- Processando imagem: {image_path} ---")
    try:
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {MODEL_WEIGHTS_PATH}")
        # Inferência crua do modelo.
        results = model(image_path)[0]
    except Exception as e:
        print(f"Erro durante a inferência na imagem {image_path}: {e}")
        continue

    # Pós-processamento com a lógica aperfeiçoada (Solidez Geometrica).
    final_detections = process_ultrasound_detections(results)

    # Visualização e salvamento para verificação.
    output_filename = f"results_v4_p{i}.png"
    visualize_results(image_path, final_detections, output_filename=output_filename)

print("\n--- Processo concluído. O motor de pós-processamento v4 com Solidez Geometrica foi aplicado. ---")
print("Os resultados do preview foram salvos. Verifique-os para confirmar que o ruído pequeno da Imagem 1 foi removido e os furos omitidos nas outras imagens foram recuperados.")
