import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union
from pythomics.models.diablo_wrapper import DIABLOWrapper
from pythomics.tools.pca import NipalsPCA

def clean_tcga_barcode(barcode: str, length: int = 12) -> str:
    """Estandariza los identificadores de muestras de TCGA."""
    clean_id = str(barcode).strip().replace('.', '-')
    if clean_id.startswith('TCGA'):
        parts = clean_id.split('-')
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return clean_id


def load_multiomics(
    data_dir: str,
    dataset_prefix: str = "KIRC",
    task_type: str = "clustering",  # Opciones: "clustering" o "classification"
    omics_files: Optional[Dict[str, str]] = None,
    survival_file: Optional[str] = None,
    labels_file: Optional[str] = None,
    impute_missing: bool = True,
    normalize_barcodes: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Carga, orienta y alinea matrices multiómicas y datos clínicos/etiquetas (incluyendo labels sin ID)."""
    
    if omics_files is None:
        omics_files = {
            'CNV': f'{dataset_prefix}_CNV.csv',
            'mRNA': f'{dataset_prefix}_mRNA.csv',
            'Methy': f'{dataset_prefix}_Methy.csv',
            'miRNA': f'{dataset_prefix}_miRNA.csv'
        }

    raw_omics = {}
    sample_sets = []
    original_sample_order = None  # Para respaldar el orden si las labels vienen sin IDs

    print(f"=== 1. Carga y Detección de Orientación [{dataset_prefix}] (Modo: {task_type.upper()}) ===")
    for omic_name, filename in omics_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo ómico: {filepath}")

        df = pd.read_csv(filepath, index_col=0)
        
        # Detección automática si las muestras están en columnas o filas
        sample_in_cols = any(str(col).startswith('TCGA') for col in df.columns)
        if sample_in_cols:
            df = df.T
            
        if normalize_barcodes:
            df.index = [clean_tcga_barcode(idx) for idx in df.index]
        
        df = df[~df.index.duplicated(keep='first')]

        raw_omics[omic_name] = df
        
        # Guardamos la lista de muestras de la primera ómica como referencia base
        if original_sample_order is None:
            original_sample_order = df.index.tolist()

        sample_sets.append(set(df.index))
        print(f"[{omic_name}] Dimensión (Muestras x Features): {df.shape}")

    # --- 2. Manejo de Supervivencia (Opcional) ---
    surv_df = None
    if survival_file is None:
        possible_surv_names = [f'survival_{dataset_prefix}.csv', f'{dataset_prefix}_survival.csv']
        for name in possible_surv_names:
            if os.path.exists(os.path.join(data_dir, name)):
                survival_file = name
                break

    if survival_file and os.path.exists(os.path.join(data_dir, survival_file)):
        surv_path = os.path.join(data_dir, survival_file)
        surv_df = pd.read_csv(surv_path, index_col=0)
        if normalize_barcodes:
            surv_df.index = [clean_tcga_barcode(idx) for idx in surv_df.index]
        surv_df = surv_df[~surv_df.index.duplicated(keep='first')]
        sample_sets.append(set(surv_df.index))
        print(f"[Survival] Registros cargados: {surv_df.shape[0]}")

    # --- 3. Manejo de Etiquetas (Soporte para formato con o sin IDs) ---
    labels_series = None
    if task_type.lower() == "classification":
        if labels_file is None:
            possible_label_names = [f'{dataset_prefix}_labels.csv', f'labels_{dataset_prefix}.csv', 'labels.csv']
            for name in possible_label_names:
                if os.path.exists(os.path.join(data_dir, name)):
                    labels_file = name
                    break
        
        if labels_file and os.path.exists(os.path.join(data_dir, labels_file)):
            labels_path = os.path.join(data_dir, labels_file)
            
            # Intentar leer asumiendo que tiene una columna de índice/IDs
            try:
                labels_df = pd.read_csv(labels_path, index_col=0)
                # Verificar si el índice parece contener barcodes de TCGA o texto válido, 
                # de lo contrario, probablemente sean solo números de fila puros (0, 1, 2...)
                first_idx_str = str(labels_df.index[0])
                if not (first_idx_str.startswith('TCGA') or '-' in first_idx_str):
                    raise ValueError("El índice detectado no parece un identificador válido.")
                
                if normalize_barcodes:
                    labels_df.index = [clean_tcga_barcode(idx) for idx in labels_df.index]
                labels_df = labels_df[~labels_df.index.duplicated(keep='first')]
                labels_series = labels_df.iloc[:, 0] if labels_df.shape[1] == 1 else labels_df
                
            except Exception:
                # Si falla o el archivo no tiene identificadores, lo leemos sin cabecera ni índice
                # y le asignamos directamente el orden de las muestras del archivo ómico base.
                labels_df = pd.read_csv(labels_path, header=0)
                if len(labels_df) != len(original_sample_order):
                    raise ValueError(
                        f"El número de filas en el archivo de labels ({len(labels_df)}) "
                        f"no coincide con el número de muestras ómicas ({len(original_sample_order)})."
                    )
                # Asignar los identificadores de la ómica base como índice del vector de etiquetas
                labels_series = pd.Series(labels_df.iloc[:, 0].values, index=original_sample_order)

            sample_sets.append(set(labels_series.index))
            print(f"[Labels] Etiquetas de clasificación cargadas exitosamente: {len(labels_series)} registros.")
        else:
            raise FileNotFoundError(f"El modo es 'classification', pero no se encontró un archivo de etiquetas válido en {data_dir}.")

    # --- 4. Intersección de Muestras Comunes ---
    print("\n=== 2. Intersección de Muestras Comunes ===")
    common_samples = sorted(list(set.intersection(*sample_sets)))
    print(f"Total de pacientes comunes alineados: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("No se encontraron muestras comunes entre los archivos. Revisa el formato de los IDs.")

    # --- 5. Filtrar y Ordenar Matrices y Metadatos ---
    aligned_omics = {}
    for omic_name, df in raw_omics.items():
        df_aligned = df.loc[common_samples].copy()
        if impute_missing and df_aligned.isnull().sum().sum() > 0:
            df_aligned = df_aligned.fillna(df_aligned.mean()).fillna(0)
        aligned_omics[omic_name] = df_aligned

    aligned_surv = surv_df.loc[common_samples].copy() if surv_df is not None else None
    aligned_labels = labels_series.loc[common_samples].copy() if labels_series is not None else None

    print("\n=== Procesamiento Completado ===")
    for omic_name, df in aligned_omics.items():
        print(f"Matriz Final {omic_name}: {df.shape}")

    return aligned_omics, aligned_surv, aligned_labels

omics, surv, y_labels = load_multiomics(
    data_dir="C:/Users/felip/MLOmics/Main_Dataset/Classification_datasets/GS-COAD/Original",
    dataset_prefix="COAD",
    task_type="classification",
    labels_file="COAD_label_num.csv" # Opcional si sigue la convención y se llama igual
)

pca = NipalsPCA(n_components=2)
#print(omics.keys())
pca.fit_transform(omics["CNV"], y_labels)
