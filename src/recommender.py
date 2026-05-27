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
DATA_PROC = ROOT / "data" 
MODELS_DIR = ROOT / "models"

# Nuevos artefactos serializados de la búsqueda por grilla
MODEL_PATHS = {
    "lowtier": MODELS_DIR / "model_low_elo.joblib",
    "apex": MODELS_DIR / "model_apex.joblib"
}
CHAMPS_PATH = DATA_PROC / "champs_metadata.csv"  # Base de datos con la Capa Gold construida

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
        # 1. Carga Dinámica de Modelos por Elos (PARCHEADO CONTRA KEYERRORS)
        self.models: Dict[str, Dict[str, Any]] = {}
        for tier, path in MODEL_PATHS.items():
            if path.exists():
                self.models[tier] = joblib.load(path)
                
                # 🛡️ Usamos .get() seguro por si el archivo .joblib de Low Elo viene sin estas llaves
                m_type = self.models[tier].get('model_type', 'xgb' if 'low' in tier else 'rf')
                m_stage = self.models[tier].get('stage', 1)
                
                print(f"✅ Cargado Backend de Inferencia para TIER: {tier.upper()} ({m_type} - Etapa {m_stage})")
            else:
                print(f"⚠️ Alerta: No se encontró el archivo del modelo para {tier} en '{path}'")

        # 2. Carga del Catálogo unificado de la Capa Gold
        self.champs_df = pd.read_csv(CHAMPS_PATH, sep=';')

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


def get_champion_roles(cid: int) -> set:
    """
    Devuelve el conjunto de roles (main y sub) que puede jugar un campeón
    para dar soporte a las composiciones FLEX.
    """
    roles = set()
    row = RES.champs_df[RES.champs_df["champ_id"] == cid]
    
    if not row.empty:
        if "main_role" in row.columns and pd.notna(row["main_role"].values[0]):
            roles.add(str(row["main_role"].values[0]).strip().upper())
        if "sub_role" in row.columns and pd.notna(row["sub_role"].values[0]):
            roles.add(str(row["sub_role"].values[0]).strip().upper())
            
    return roles if roles else {"FLEX"}


def role_penalty(cand_id: int, current_ids: List[int]) -> float:
    """
    Aplica penalizaciones dinámicas por redundancia estructural 
    permitiendo transiciones de campeones Flex (ej: Karma Sup + Orianna Mid).
    """
    cand_roles = get_champion_roles(cand_id)
    if "FLEX" in cand_roles: 
        return 1.0

    # Obtenemos los sets de roles de todos los campeones ya elegidos en la escuadra
    team_champions_roles = [get_champion_roles(cid) for cid in current_ids if cid != -1]
    
    # Evaluamos el peor solapamiento de roles para el candidato
    max_overlap = 0
    for cand_role in cand_roles:
        # Contamos cuántos personajes en el equipo ya reclaman o pueden jugar este rol
        count_same = sum(cand_role in roles for roles in team_champions_roles)
        if count_same > max_overlap:
            max_overlap = count_same

    # Escala inteligente de penalización flex
    if max_overlap == 0: return 1.0
    if max_overlap == 1: return 0.95  # Penalización casi nula (permite el pick si hay un flex en mesa)
    if max_overlap == 2: return 0.60  # Alerta: posible triple rol idéntico innecesario
    return 0.25


# ==========================================
# 🧬 Tubería de Construcción de Características
# ==========================================

def build_features_for_draft(blue_ids, red_ids, tier="lowtier"):
    """
    Construye el vector de características para los modelos predictivos.
    Dado que ambos modelos (Apex y Low Elo) usan las mismas 13 variables,
    se unifica el retorno para evitar descuadres en Pandas.
    """
    tier_key = tier.lower()
    
    # 1. Componente base (5 IDs Blue + 5 IDs Red = 10 columnas)
    feat_data = []
    feat_data.extend(blue_ids)
    feat_data.extend(red_ids)
    
    # 2. Cálculo de Win Rates del servidor según el Elo seleccionado
    wr_col = f"win_rate_{tier_key}" if f"win_rate_{tier_key}" in RES.champs_df.columns else "win_rate_role"
    wr_map = RES.champs_df.set_index("champ_id")[wr_col].to_dict()
    
    b_wrs = [wr_map.get(c, 0.50) for c in blue_ids if c != -1]
    r_wrs = [wr_map.get(c, 0.50) for c in red_ids if c != -1]
    
    b_mean = np.mean(b_wrs) if b_wrs else 0.50
    r_mean = np.mean(r_wrs) if r_wrs else 0.50
    d_wr = b_mean - r_mean
    
    # Añadimos las 3 métricas macro (+3 columnas = 13 en total)
    feat_data.extend([b_mean, r_mean, d_wr])
    
    # 3. Definición estricta de los 13 nombres de columnas esenciales
    # IMPORTANTE: Este orden debe ser idéntico al que usaste en tu Jupyter Notebook / script de entrenamiento
    feature_names = [f"b{i}" for i in range(1, 6)] + [f"r{i}" for i in range(1, 6)] + ["b_mean", "r_mean", "d_wr"]
    
    # Retornamos el DataFrame limpio con las 13 columnas simétricas para AMBOS modelos
    return pd.DataFrame([feat_data], columns=feature_names)


# ==========================================
# 🧠 Motores de Inferencia y Explicabilidad
# ==========================================

def predict_blue_win_prob(df_feats, tier="lowtier"):
    """
    Ejecuta la predicción de victoria detectando dinámicamente la librería 
    del modelo (XGBoost vs Scikit-Learn) sin depender de llaves de texto.
    """
    tier_key = tier.lower()
    model_entry = RES.models.get(tier_key)
    
    if not model_entry:
        return 0.50 # Fallback neutral si el modelo no existe

    # Extraemos el objeto de machine learning real de forma segura
    if isinstance(model_entry, dict):
        model = model_entry.get("model", model_entry)
    else:
        model = model_entry

    # 🧠 DETECCIÓN DINÁMICA: Analizamos el nombre de la clase real del objeto
    class_name = type(model).__name__.lower()
    is_xgb = "xgb" in class_name or "booster" in class_name

    # =========================================================================
    # 🔰 SI EL MODELO ES XGBOOST (Low Elo)
    # =========================================================================
    if is_xgb:
        try:
            # Si se guardó usando la API compatible con Sklearn (XGBClassifier)
            preds_proba = model.predict_proba(df_feats)
            return float(preds_proba[0][1])
        except AttributeError:
            # Si se guardó usando la API nativa de XGBoost (xgb.train)
            import xgboost as xgb
            dmat = xgb.DMatrix(df_feats)
            preds = model.predict(dmat)
            return float(preds[0])

    # =========================================================================
    # 🏆 SI EL MODELO ES RANDOM FOREST / SKLEARN (Apex)
    # =========================================================================
    else:
        preds_proba = model.predict_proba(df_feats)
        return float(preds_proba[0][1])

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
    top_k: int = 5,
    bans: List[Union[int, str]] = None
) -> List[Recommendation]:
    """
    Evalúa el catálogo total disponible frente al estado actual del Draft.
    Usa la función predict_blue_win_prob unificada para evitar KeyErrors.
    """
    tier_key = tier.lower()
    if tier_key not in RES.models:
        return []

    # 1. Inicializar y normalizar estados del Draft
    blue_ids = [normalize_champion(c) for c in blue_champs]
    red_ids = [normalize_champion(c) for c in red_champs]
    ban_ids = [normalize_champion(c) for c in bans] if bans else []
    
    # Excluir de las recomendaciones lo que ya está pickeado o baneado
    used_ids = set(blue_ids + red_ids + ban_ids)

    # Mapeo rápido de IDs a Nombres para las explicaciones
    id_to_name = RES.champs_df.set_index("champ_id")["name"].to_dict()

    # =========================================================================
    # Estado Inicial Vacío (Top 5 Win Rate base del Tier)
    # =========================================================================
    if len(blue_ids) == 0 and len(red_ids) == 0:
        wr_col = f"win_rate_{tier_key}" if f"win_rate_{tier_key}" in RES.champs_df.columns else "win_rate_role"
        available_champs = RES.champs_df[~RES.champs_df["champ_id"].isin(used_ids)]
        top_champs = available_champs.sort_values(by=wr_col, ascending=False).head(top_k)
        
        return [
            Recommendation(
                champ_id=int(row["champ_id"]),
                champ_name=row["name"],
                prob_blue_win=float(row[wr_col]) if side == "blue" else 1.0 - float(row[wr_col]),
                prob_red_win=float(row[wr_col]) if side == "red" else 1.0 - float(row[wr_col]),
                score=float(row[wr_col]),
                explanation=f"Mayor Win Rate base en este Elo ({float(row[wr_col])*100:.1f}%)."
            )
            for _, row in top_champs.iterrows()
        ]

    # =========================================================================
    # Evaluación de Candidatos en tiempo real
    # =========================================================================
    candidate_ids = [cid for cid in RES.champs_df["champ_id"].astype(int) if cid not in used_ids]
    results = []

    for cid in candidate_ids:
        # Clonamos el estado actual para simular el escenario con este campeón
        sim_blue = list(blue_ids)
        sim_red = list(red_ids)
        
        if side == "blue":
            sim_blue.append(cid)
        else:
            sim_red.append(cid)
            
        # Rellenamos con -1 para mantener la estructura simétrica de 13 columnas de features
        while len(sim_blue) < 5: sim_blue.append(-1)
        while len(sim_red) < 5: sim_red.append(-1)
        
        # Construimos las características
        df_feats = build_features_for_draft(sim_blue, sim_red, tier=tier_key)
        
        # ✨ EL FIX: Delegar la predicción a la función blindada que ya creamos
        try:
            p_blue = predict_blue_win_prob(df_feats, tier=tier_key)
            p_red = 1.0 - p_blue
        except Exception:
            # Fallback de seguridad por si falla el modelo en caliente
            p_blue, p_red = 0.50, 0.50

        # Determinamos la probabilidad objetivo según el bando que consulta
        target_prob = p_blue if side == "blue" else p_red
        
        # Calculamos la penalización por redundancia de roles (Flex System)
        current_team_ids = blue_ids if side == "blue" else red_ids
        penalty = role_penalty(cid, current_team_ids)
        
        # Score Táctico final ponderado
        score_tactico = target_prob * penalty
        
        # Extraemos etiquetas de texto para las explicaciones contextuales
        champ_name = id_to_name.get(cid, f"Champ {cid}")
        row_meta = RES.champs_df[RES.champs_df["champ_id"] == cid]
        
        explanation = "Pick sólido para balancear la composición."
        if not row_meta.empty and "tactic_tags" in row_meta.columns:
            tags = row_meta["tactic_tags"].values[0]
            if pd.notna(tags) and str(tags).strip():
                explanation = str(tags)

        results.append(
            Recommendation(
                champ_id=cid,
                champ_name=champ_name,
                prob_blue_win=p_blue,
                prob_red_win=p_red,
                score=score_tactico,
                explanation=explanation
            )
        )

    # Ordenamos de mayor a menor según el Score Táctico resultante
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


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
