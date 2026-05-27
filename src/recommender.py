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
        # 1. Carga Dinámica de Modelos por Elos
        self.models: Dict[str, Any] = {}
        for tier, path in MODEL_PATHS.items():
            if path.exists():
                loaded_object = joblib.load(path)
                self.models[tier] = loaded_object
                
                # 🛡️ Verificación de tipo: ¿Es un diccionario o un Pipeline/Modelo directo?
                if isinstance(loaded_object, dict):
                    m_type = loaded_object.get('model_type', 'rf')
                    m_stage = loaded_object.get('stage', 1)
                else:
                    # Si es un Pipeline directo (como tu Low Elo)
                    class_name = type(loaded_object).__name__.lower()
                    m_type = 'xgb_pipeline' if 'xgb' in class_name or 'pipeline' in class_name else 'model'
                    m_stage = 2  # Asumimos etapa avanzada por ser Pipeline

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

        # Orden riguroso de las 9 dimensiones de la Capa Gold Lvl 13
        self.gold_cols = [
            'Gold_Phys_Dmg', 'Gold_Mag_Dmg', 'Gold_DPS', 
            'Gold_Durability', 'Gold_CC', 'Gold_Poke', 
            'Gold_Engage', 'Gold_Utility', 'Gold_Kiting'
        ]

        # Construcción de vectores de consulta rápida
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
    Apex -> Requiere 13 columnas macros.
    Low Elo -> El StandardScaler del Pipeline exige estrictamente 26 columnas.
    """
    tier_key = tier.lower()
    
    # 1. Componente base común (10 columnas: IDs de campeones)
    feat_data = []
    feat_data.extend(blue_ids)
    feat_data.extend(red_ids)
    
    # 2. Win Rates macros (3 columnas: b_mean, r_mean, d_wr)
    wr_col = f"win_rate_{tier_key}" if f"win_rate_{tier_key}" in RES.champs_df.columns else "win_rate_role"
    wr_map = RES.champs_df.set_index("champ_id")[wr_col].to_dict()
    
    b_wrs = [wr_map.get(c, 0.50) for c in blue_ids if c != -1]
    r_wrs = [wr_map.get(c, 0.50) for c in red_ids if c != -1]
    
    b_mean = np.mean(b_wrs) if b_wrs else 0.50
    r_mean = np.mean(r_wrs) if r_wrs else 0.50
    d_wr = b_mean - r_mean
    
    feat_data.extend([b_mean, r_mean, d_wr]) # Aquí acumulamos 13 elementos
    
    # Nombres base para las primeras 13 columnas
    feature_names = [f"b{i}" for i in range(1, 6)] + [f"r{i}" for i in range(1, 6)] + ["b_mean", "r_mean", "d_wr"]
    
    # =========================================================================
    # 🔰 CASO LOW ELO: Expandir a 26 columnas para satisfacer al StandardScaler
    # =========================================================================
    if tier_key == "lowtier":
        # Extraemos los deltas dimensionales de la Capa Gold (9 columnas)
        blue_clean = [c for c in blue_ids if c != -1]
        red_clean = [c for c in red_ids if c != -1]
        
        b_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in blue_clean], axis=0) if blue_clean else np.zeros(9)
        r_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in red_clean], axis=0) if red_clean else np.zeros(9)
        
        deltas_9 = b_v - r_v
        feat_data.extend(deltas_9) # 13 + 9 = 22 elementos
        
        # Añadimos los nombres de las columnas de la Capa Gold
        for col_name in RES.gold_cols:
            feature_names.append(f"delta_{col_name}")
            
        # 🛡️ RELLENO DE SEGURIDAD: Agregamos placeholders en 0.0 hasta completar las 26 exigidas
        while len(feat_data) < 26:
            feat_data.append(0.0)
            feature_names.append(f"placeholder_{len(feat_data)}")
            
    # Retorna un DataFrame con 13 columnas para Apex o 26 exactas para Low Elo
    return pd.DataFrame([feat_data], columns=feature_names)

# ==========================================
# 🧠 Motores de Inferencia y Explicabilidad
# ==========================================

def predict_blue_win_prob(df_feats, tier="lowtier"):
    """
    Predice la probabilidad de victoria adaptando dinámicamente las columnas
    de entrada según el número exacto de variables que el modelo espera (1, 13 o 26).
    """
    tier_key = tier.lower()
    model_entry = RES.models.get(tier_key)
    
    if model_entry is None:
        return 0.50

    # 1. Extraer el objeto del modelo real
    if isinstance(model_entry, dict):
        model = model_entry.get("model")
        if model is None:
            for val in model_entry.values():
                if hasattr(val, "predict_proba") or hasattr(val, "predict"):
                    model = val
                    break
            if model is None:
                model = model_entry
    else:
        model = model_entry

    # =========================================================================
    # 🧠 ADAPTACIÓN DINÁMICA DE ENTRADA (Filtro Anti-Descuadres)
    # =========================================================================
    X_input = df_feats
    
    if hasattr(model, "n_features_in_"):
        n_expected = model.n_features_in_
        
        if n_expected == 1:
            # ✨ EL FIX: Si el modelo solo espera 1 variable (ej: d_wr), filtramos solo esa columna
            if "d_wr" in df_feats.columns:
                X_input = df_feats[["d_wr"]]
            else:
                X_input = df_feats.iloc[:, [-1]] # Fallback: Tomar la última columna
                
        elif n_expected == 13:
            # Asegurar que tenga estrictamente las 13 columnas macros base
            base_cols = [f"b{i}" for i in range(1, 6)] + [f"r{i}" for i in range(1, 6)] + ["b_mean", "r_mean", "d_wr"]
            X_input = df_feats[[c for c in base_cols if c in df_feats.columns]]
            if X_input.shape[1] != 13:
                X_input = df_feats.iloc[:, :13]

    # =========================================================================
    # ⚙️ Ejecución de la Predicción
    # =========================================================================
    try:
        preds_proba = model.predict_proba(X_input)
        return float(preds_proba[0][1])
    except AttributeError:
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(X_input) if not isinstance(X_input, xgb.DMatrix) else X_input
            preds = model.predict(dmat)
            return float(preds[0])
        except Exception:
            return 0.50

    # =========================================================================
    # ⚙️ Ejecución Segura de la Predicción
    # =========================================================================
    try:
        # Tanto el Pipeline de Sklearn como el XGBClassifier usan predict_proba
        preds_proba = model.predict_proba(df_feats)
        return float(preds_proba[0][1])
    except AttributeError:
        # Fallback si el modelo es un Booster nativo puro de XGBoost
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(df_feats) if not isinstance(df_feats, xgb.DMatrix) else df_feats
            preds = model.predict(dmat)
            return float(preds[0])
        except Exception:
            # Última línea de defensa analítica ante cualquier imprevisto
            return 0.50
    # =========================================================================
    # ⚙️ Ejecución de la Predicción
    # =========================================================================
    try:
        # Tanto el Pipeline de Sklearn como el XGBClassifier usan predict_proba
        preds_proba = model.predict_proba(df_feats)
        return float(preds_proba[0][1])
    except AttributeError:
        # Fallback de seguridad extrema por si es un Booster nativo puro de XGBoost
        import xgboost as xgb
        dmat = xgb.DMatrix(df_feats)
        preds = model.predict(dmat)
        return float(preds[0])

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
    # Estado Inicial Vacío (Top 5 Win Rate base del Tier) - CORREGIDO
    # =========================================================================
    if len(blue_ids) == 0 and len(red_ids) == 0:
        wr_col = f"win_rate_{tier_key}" if f"win_rate_{tier_key}" in RES.champs_df.columns else "win_rate_role"
        available_champs = RES.champs_df[~RES.champs_df["champ_id"].isin(used_ids)]
        top_champs = available_champs.sort_values(by=wr_col, ascending=False).head(top_k)
        
        recs = []
        for _, row in top_champs.iterrows():
            # 🛡️ Si el win rate viene como 55.61, lo normalizamos a 0.5561
            raw_wr = float(row[wr_col])
            base_prob = raw_wr / 100.0 if raw_wr > 1.0 else raw_wr
            
            p_blue = base_prob if side == "blue" else 1.0 - base_prob
            p_red = base_prob if side == "red" else 1.0 - base_prob
            
            recs.append(
                Recommendation(
                    champ_id=int(row["champ_id"]),
                    champ_name=row["name"],
                    prob_blue_win=p_blue,
                    prob_red_win=p_red,
                    score=base_prob, # El score táctico ahora será 0.556 en vez de 55.6
                    explanation=f"Mayor Win Rate base en este Elo ({raw_wr if raw_wr > 1.0 else raw_wr*100:.1f}%)."
                )
            )
        return recs

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
