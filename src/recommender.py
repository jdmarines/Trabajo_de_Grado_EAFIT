"""
Motor de recomendación de picks optimizado para el draft de LoL (Capa Gold & Multi-Tier).

Dado un estado parcial de composición, simula añadir cada campeón posible y utiliza 
el pipeline optimizado del Tier correspondiente para estimar el impacto en la probabilidad de victoria.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import joblib


# ==========================================
# 📌 Configuración de Rutas y Constantes
# ==========================================

ROOT = Path(__file__).resolve().parents[1]  # /workspaces/LOL_Draft_Assistant
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Nuevos artefactos serializados de la búsqueda por grilla
MODEL_PATHS = {
    "lowtier": MODELS_DIR / "best_model_lowtier.joblib",
    "apex": MODELS_DIR / "best_model_apex.joblib"
}
CHAMPS_PATH = DATA_PROC / "champs_gold_lvl13.csv"  # Base de datos con la Capa Gold construida


# ==========================================
# 📊 Data Classes para Salida Estructurada
# ==========================================

@dataclass
class Recommendation:
    champ_id: int
    champ_name: str
    prob_blue_win: float
    prob_red_win: float
    score: float
    explanation: str


# ==========================================
# 📦 Administrador de Recursos (Modelos y Datos)
# ==========================================

class Resources:
    def __init__(self):
        # 1. Carga Dinámica de Modelos por Elos
        self.models: Dict[str, Dict[str, Any]] = {}
        for tier, path in MODEL_PATHS.items():
            if path.exists():
                self.models[tier] = joblib.load(path)
                print(f"✅ Cargado Backend de Inferencia para TIER: {tier.upper()} ({self.models[tier]['model_type']} - Etapa {self.models[tier]['stage']})")
            else:
                print(f"⚠️ Alerta: No se encontró el archivo del modelo para {tier} en '{path}'")

        # 2. Carga del Catálogo unificado de la Capa Gold
        self.champs_df = pd.read_csv(CHAMPS_PATH)

        # Mapeos de Identidad Indispensables
        self.id2name: Dict[int, str] = self.champs_df.set_index("champ_id")["name"].to_dict()
        
        self.name2id: Dict[str, int] = {}
        for _, row in self.champs_df.iterrows():
            cid = int(row["champ_id"])
            self.name2id[str(row["name"]).strip().lower()] = cid
            if "apiname" in row and pd.notna(row["apiname"]):
                self.name2id[str(row["apiname"]).strip().lower()] = cid

        # Orden riguroso e invariante de las 9 dimensiones de la Capa Gold Lvl 13
        self.gold_cols = [
            'Gold_Phys_Dmg', 'Gold_Mag_Dmg', 'Gold_DPS', 
            'Gold_Durability', 'Gold_CC', 'Gold_Poke', 
            'Gold_Engage', 'Gold_Utility', 'Gold_Kiting'
        ]

        # Construcción de tensores de consulta rápida por campeón
        self.champ_vectors: Dict[int, np.ndarray] = {}
        for _, row in self.champs_df.iterrows():
            cid = int(row["champ_id"])
            self.champ_vectors[cid] = np.array([row[col] for col in self.gold_cols], dtype=float)


RES = Resources()


# ==========================================
# 🛠️ Utilidades de Procesamiento y Filtrado
# ==========================================

def normalize_champion(spec: Union[int, str]) -> int:
    if isinstance(spec, int) or (isinstance(spec, float) and spec.is_integer()):
        return int(spec)
    key = str(spec).strip().lower()
    if key not in RES.name2id:
        raise ValueError(f"El campeón '{spec}' no fue encontrado en la base de datos de la Capa Gold.")
    return RES.name2id[key]


def get_primary_role(cid: int) -> str:
    if "main_role" in RES.champs_df.columns:
        return str(RES.champs_df.set_index("champ_id").at[cid, "main_role"])
    return "FLEX"


def role_penalty(cand_id: int, current_ids: List[int]) -> float:
    """Aplica penalizaciones por redundancia estructural en el equipo (ej: triple ADC)."""
    cand_role = get_primary_role(cand_id)
    if cand_role == "FLEX":
        return 1.0

    roles = [get_primary_role(cid) for cid in current_ids if cid != -1]
    count_same = sum(r == cand_role for r in roles)

    if count_same == 0: return 1.0
    if count_same == 1: return 0.85  # Penalización ligera para duplicados válidos (ej: doble tanque)
    if count_same == 2: return 0.50  # Composición desbalanceada
    return 0.20                      # Forzado incomprensible estructuralmente


# ==========================================
# 🧬 Tubería de Construcción de Características
# ==========================================

def build_features_for_draft(blue_ids: List[int], red_ids: List[int], tier: str) -> pd.DataFrame:
    """
    Simula de forma exacta el comportamiento de la función 'get_stage_data' de entrenamiento.
    Reconstruye las variables correspondientes a la etapa óptima guardada en el artefacto.
    """
    tier_key = tier.lower()
    if tier_key not in RES.models:
        raise ValueError(f"El backend de inferencia para el tier '{tier}' no está inicializado.")

    artifact = RES.models[tier_key]
    stage = artifact['stage']
    expected_features = artifact['feature_names']

    # 1. Componente Macro: Extracción y Diferencial de Win Rate por Elo
    wr_col = f"win_rate_{tier_key}" if f"win_rate_{tier_key}" in RES.champs_df.columns else "win_rate_role"
    wr_map = RES.champs_df.set_index("champ_id")[wr_col].to_dict()

    b_wrs = [wr_map.get(c, 0.50) for c in blue_ids if c != -1]
    r_wrs = [wr_map.get(c, 0.50) for c in red_ids if c != -1]

    d_wr = np.mean(b_wrs) - np.mean(r_wrs) if b_wrs and r_wrs else 0.0

    # 2. Componente Micro Táctico: Sumatorios de Vectores Gold (Placeholders -1 aportan ceros)
    b_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in blue_ids], axis=0)
    r_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in red_ids], axis=0)

    deltas_22 = list(b_v - r_v)

    # 3. Componente Dinámico: Cómputo de Ratios Relacionales Cruzados
    # Índices: 0:Phys_Dmg, 2:DPS, 3:Durability, 4:CC, 5:Poke, 6:Engage
    r1 = (b_v[0] / (r_v[3] + 1)) - (r_v[0] / (b_v[3] + 1))  # R1: Ráfaga Física vs Puntos de Salud Efectivos
    r2 = (b_v[5] / (r_v[6] + 1)) - (r_v[5] / (b_v[6] + 1))  # R2: Hostigamiento en Rango vs Iniciación Macro
    r3 = (b_v[4] / (r_v[2] + 1)) - (r_v[4] / (b_v[2] + 1))  # R3: Densidad de CC vs Capacidad de Daño por Segundo
    ratios = [r1, r2, r3]

    # 4. Enrutamiento Estricto según la Etapa del Artefacto
    if stage == 1:
        f = [d_wr]
    elif stage == 2:
        f = deltas_22
    elif stage == 3:
        f = deltas_22 + ratios
    elif stage == 4:
        f = [d_wr] + deltas_22 + ratios
    else:
        raise ValueError(f"Etapa de diseño operacional {stage} desconocida.")

    # Convertimos a DataFrame asignando las columnas exactas del entrenamiento para el Pipeline
    return pd.DataFrame([f], columns=expected_features)


# ==========================================
# 🧠 Motores de Inferencia y Explicabilidad
# ==========================================

def predict_blue_win_prob(df_features: pd.DataFrame, tier: str) -> float:
    pipeline = RES.models[tier.lower()]['pipeline']
    return float(pipeline.predict_proba(df_features)[0, 1])


def explain_candidate(champ_id: int, side: str) -> str:
    """Genera explicaciones cualitativas diagnósticas basadas en el perfil Gold del personaje."""
    vec = RES.champ_vectors.get(champ_id, np.zeros(9))
    factors = []

    # Mapeo semántico de los aportes de la capa Gold a Nivel 13
    if vec[4] > 3.5: factors.append("Alta Densidad de CC para Batallas Grupales")
    if vec[6] > 2.8: factors.append("Excelente Capacidad de Iniciación Forzada (Engage)")
    if vec[3] > 4.5: factors.append("Alta Absorción de Castigo (Salud Efectiva)")
    if vec[5] > 6.5: factors.append("Alto Potencial de Desgaste de Larga Distancia (Poke)")
    if vec[7] > 2.2: factors.append("Fuerte Mitigación de Errores Aliados (Utility)")
    if vec[8] > 5.5: factors.append("Excelente Perfil de Espaciado y Retroceso (Kiting)")
    
    if vec[0] > vec[1] and vec[0] > 600: factors.append("Ráfaga Física Dominante")
    elif vec[1] > vec[0] and vec[1] > 600: factors.append("Aporte de Presión de Daño Mágico")

    if not factors:
        return "Ajuste de balance posicional y Metajuego."
    return " | ".join(factors[:2])


# ==========================================
# 🏆 Algoritmo de Recomendación Secuencial
# ==========================================

def recommend_for(
    blue_champs: List[Union[int, str]],
    red_champs: List[Union[int, str]],
    side: str = "blue",
    tier: str = "lowtier",
    top_k: int = 5
) -> List[Recommendation]:
    """
    Evalúa el catálogo total disponible frente al estado actual del Draft.
    Devuelve una lista ordenada por Score táctico (Probabilidad × Penalización de Rol).
    """
    tier_key = tier.lower()
    if tier_key not in RES.models:
        return []

    # 1. Inicializar y normalizar estados del Draft
    blue_ids = [normalize_champion(c) for c in blue_champs]
    red_ids = [normalize_champion(c) for c in red_champs]
    used_ids = set(blue_ids + red_ids)

    # Identificar el universo de campeones elegibles en el parche
    candidate_ids = [cid for cid in RES.champs_df["champ_id"].astype(int) if cid not in used_ids]

    # Construcción de la Línea Base (Baseline del estado actual del Draft)
    base_blue = blue_ids.copy()
    base_red = red_ids.copy()
    while len(base_blue) < 5: base_blue.append(-1)
    while len(base_red) < 5: base_red.append(-1)

    recs: List[Recommendation] = []
    side_key = side.lower()

    # 2. Bucle de Simulación Secuencial Turno a Turno
    for cid in candidate_ids:
        if side_key == "blue":
            sim_blue = blue_ids + [cid]
            sim_red = red_ids
            current_team_ids = blue_ids
        else:
            sim_blue = blue_ids
            sim_red = red_ids + [cid]
            current_team_ids = red_ids

        # Rellenar con placeholders neutrales para cumplir las dimensiones estructurales
        while len(sim_blue) < 5: sim_blue.append(-1)
        while len(sim_red) < 5: sim_red.append(-1)

        try:
            # Construir variables adaptadas al clasificador asignado
            df_feats = build_features_for_draft(sim_blue, sim_red, tier_key)
            
            # Obtener inferencia probabilística limpia del Pipeline
            p_blue = predict_blue_win_prob(df_feats, tier_key)
            p_red = 1.0 - p_blue

            # El score base del bando objetivo corresponde a su propia probabilidad de victoria
            base_score = p_blue if side_key == "blue" else p_red
            
            # Aplicar la penalización de balance de roles en la composición grupal
            pen = role_penalty(cid, current_team_ids)
            final_score = base_score * pen

            recs.append(
                Recommendation(
                    champ_id=cid,
                    champ_name=RES.id2name.get(cid, f"Desconocido_{cid}"),
                    prob_blue_win=p_blue,
                    prob_red_win=p_red,
                    score=final_score,
                    explanation=explain_candidate(cid, side_key)
                )
            )
        except Exception as e:
            continue

    # 3. Ordenamiento jerárquico descendente basado en el Score Estratégico
    recs_sorted = sorted(recs, key=lambda r: r.score, reverse=True)
    return recs_sorted[:top_k]


# ==========================================
# 🧪 Pruebas de Diagnóstico por Consola (CLI)
# ==========================================

if __name__ == "__main__":
    # Simulación de un Draft en progreso
    comp_azul = ["Orianna", "Jinx"]
    comp_roja = ["Malphite", "Thresh"]

    print("--- 🔵 SIMULACIÓN TURNOS LOWTIER (Etapa 4 Real Time) ---")
    recs_low = recommend_for(comp_azul, comp_roja, side="blue", tier="lowtier", top_k=3)
    for idx, r in enumerate(recs_low, 1):
        print(f"{idx}. {r.champ_name:14s} | Score: {r.score:.4f} | P(Azul): {r.prob_blue_win:.3f} | Diagnóstico: {r.explanation}")

    print("\n--- 🔴 SIMULACIÓN TURNOS APEX (Etapa 1 Pure Meta) ---")
    recs_apex = recommend_for(comp_azul, comp_roja, side="red", tier="apex", top_k=3)
    for idx, r in enumerate(recs_apex, 1):
        print(f"{idx}. {r.champ_name:14s} | Score: {r.score:.4f} | P(Rojo): {r.prob_red_win:.3f} | Diagnóstico: {r.explanation}")
