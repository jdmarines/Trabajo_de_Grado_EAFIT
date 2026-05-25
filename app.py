import streamlit as st
from pathlib import Path
import sys
import numpy as np

# =====================================
# CONFIG Y IMPORTS
# =====================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

# Importamos los componentes del nuevo motor optimizado
from recommender import (
    RES,
    normalize_champion,
    build_features_for_draft,
    predict_blue_win_prob,
    recommend_for,
)

st.set_page_config(
    page_title="LoL Draft Recommender Pro",
    page_icon="🎮",
    layout="wide",
)

# =====================================
# UTILIDADES Y PROCESAMIENTO TÁCTICO
# =====================================

def normalize_selection(selection_list):
    """Quita '(vacío)' y devuelve solo los nombres de campeones."""
    return [c for c in selection_list if c != "(vacío)"]

def calculate_gold_deltas(blue_sel, red_sel):
    """
    Calcula los diferenciales de la Capa Gold expresados en 
    porcentaje de ventaja relativa (-100% a +100%).
    """
    blue_ids = [normalize_champion(c) for c in blue_sel]
    red_ids = [normalize_champion(c) for c in red_sel]
    
    # Sumar vectores de la capa Gold por equipo
    b_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in blue_ids], axis=0)
    r_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in red_ids], axis=0)
    
    # Calculamos la base total de la mesa para cada métrica
    total_v = b_v + r_v
    
    # Evitamos la división por cero: si el total es 0, la ventaja es 0.0%
    delta_pct = np.where(total_v > 0, ((b_v - r_v) / total_v) * 100, 0.0)
    
    return {
        "phys_dmg": delta_pct[0],
        "mag_dmg": delta_pct[1],
        "dps": delta_pct[2],
        "durability": delta_pct[3],
        "cc": delta_pct[4],
        "poke": delta_pct[5],
        "engage": delta_pct[6],
        "utility": delta_pct[7],
        "kiting": delta_pct[8]
    }

def coach_summary(m: dict) -> str:
    """
    Genera un diagnóstico estratégico basado en las ventajas porcentuales.
    """
    lines = []

    # Control de Masas (CC)
    if m["cc"] > 10.0:
        lines.append("BLUE tiene **mayor densidad de CC (+)**, ideal para asegurar cazadas y neutralizar objetivos clave.")
    elif m["cc"] < -10.0:
        lines.append("RED cuenta con **ventaja notable en CC**, BLUE debe cuidar sus líneas de entrada y usar limpiar/tenacidad.")

    # Iniciación (Engage)
    if m["engage"] > 10.0:
        lines.append("La composición BLUE cuenta con **mejor engage directo**, manteniendo la iniciativa para proponer peleas.")
    elif m["engage"] < -10.0:
        lines.append("BLUE está **en desventaja de iniciación**; es crucial jugar al counter-engage o desenganche rápido.")

    # Durabilidad (Frontline)
    if m["durability"] > 12.0:
        lines.append("BLUE posee una **frontline más sólida y robusta**, óptima para combates grupales extendidos.")
    elif m["durability"] < -12.0:
        lines.append("RED es **significativamente más resistente**, BLUE debe evitar choques frontales y buscar flancos.")

    # Desgaste y Espaciado (Poke & Kiting)
    if m["poke"] > 10.0:
        lines.append("BLUE destaca en **capacidad de Poke**, busque debilitar al rival antes de iniciar los objetivos neutrales.")
    if m["kiting"] > 10.0:
        lines.append("BLUE tiene **mejor perfil de Kiting**, lo que le permite estirar las peleas y castigar mientras retrocede.")

    # Balance y Sesgo de Daño
    if m["phys_dmg"] > 20.0 and m["mag_dmg"] < -20.0:
        lines.append("La composición de BLUE está **fuertemente sesgada hacia el daño físico**, facilitando que RED acumule armadura.")
    elif m["mag_dmg"] > 20.0 and m["phys_dmg"] < -20.0:
        lines.append("BLUE ejerce una **presión masiva de daño mágico**, obligando a RED a priorizar resistencia mágica temprana.")

    if not lines:
        return "Las composiciones se encuentran altamente equilibradas en sus ejes tácticos principales."

    return " ".join(lines)


# =====================================
# INTERFAZ MODULAR DE DRAFT
# =====================================

def render_draft_interface(tier_key):
    """
    Renderiza el simulador de draft para un tier específico ('apex' o 'lowtier').
    Incluye fase de bloqueos (Bans) y filtros estrictos anti-duplicados.
    """
    # Catálogo unificado obtenido del dataframe cargado en el backend
    champ_list = sorted(RES.champs_df["name"].tolist())

    # 🛠️ FUNCIÓN AUXILIAR: Filtra campeones seleccionados O baneados en cualquier casilla
    def get_available_options(current_key):
        # Registramos las llaves de los picks y también de los bans
        all_keys = (
            [f"b{i}_{tier_key}" for i in range(1, 6)] + 
            [f"r{i}_{tier_key}" for i in range(1, 6)] +
            [f"bb{i}_{tier_key}" for i in range(1, 6)] + 
            [f"rb{i}_{tier_key}" for i in range(1, 6)]
        )
        selected_elsewhere = set()
        
        for k in all_keys:
            if k != current_key:
                val = st.session_state.get(k, "(vacío)")
                if val != "(vacío)":
                    selected_elsewhere.add(val)
                    
        return ["(vacío)"] + [c for c in champ_list if c not in selected_elsewhere]

    # =========================================================================
    # 🚫 FASE DE BLOQUEOS (BANS) - Interfaz Colapsable Compacta
    # =========================================================================
    with st.expander("🚫 Fase de Bloqueos (Bans)", expanded=False):
        b_col, r_col = st.columns(2)
        
        with b_col:
            st.markdown("<span style='color:#4A90E2'>**Bans Equipo BLUE**</span>", unsafe_allow_html=True)
            bb_cols = st.columns(5)
            bb1 = bb_cols[0].selectbox("Ban 1", get_available_options(f"bb1_{tier_key}"), key=f"bb1_{tier_key}", label_visibility="collapsed")
            bb2 = bb_cols[1].selectbox("Ban 2", get_available_options(f"bb2_{tier_key}"), key=f"bb2_{tier_key}", label_visibility="collapsed")
            bb3 = bb_cols[2].selectbox("Ban 3", get_available_options(f"bb3_{tier_key}"), key=f"bb3_{tier_key}", label_visibility="collapsed")
            bb4 = bb_cols[3].selectbox("Ban 4", get_available_options(f"bb4_{tier_key}"), key=f"bb4_{tier_key}", label_visibility="collapsed")
            bb5 = bb_cols[4].selectbox("Ban 5", get_available_options(f"bb5_{tier_key}"), key=f"bb5_{tier_key}", label_visibility="collapsed")
            
        with r_col:
            st.markdown("<span style='color:#E03A3E'>**Bans Equipo RED**</span>", unsafe_allow_html=True)
            rb_cols = st.columns(5)
            rb1 = rb_cols[0].selectbox("Ban 1", get_available_options(f"rb1_{tier_key}"), key=f"rb1_{tier_key}", label_visibility="collapsed")
            rb2 = rb_cols[1].selectbox("Ban 2", get_available_options(f"rb2_{tier_key}"), key=f"rb2_{tier_key}", label_visibility="collapsed")
            rb3 = rb_cols[2].selectbox("Ban 3", get_available_options(f"rb3_{tier_key}"), key=f"rb3_{tier_key}", label_visibility="collapsed")
            rb4 = rb_cols[3].selectbox("Ban 4", get_available_options(f"rb4_{tier_key}"), key=f"rb4_{tier_key}", label_visibility="collapsed")
            rb5 = rb_cols[4].selectbox("Ban 5", get_available_options(f"rb5_{tier_key}"), key=f"rb5_{tier_key}", label_visibility="collapsed")

    # Guardamos la lista de bloqueos activos
    ban_sel = normalize_selection([bb1, bb2, bb3, bb4, bb5, rb1, rb2, rb3, rb4, rb5])

    # =========================================================================
    # ⚔️ FASE DE SELECCIÓN (PICKS)
    # =========================================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Equipo BLUE")
        b1 = st.selectbox("Blue 1", get_available_options(f"b1_{tier_key}"), key=f"b1_{tier_key}")
        b2 = st.selectbox("Blue 2", get_available_options(f"b2_{tier_key}"), key=f"b2_{tier_key}")
        b3 = st.selectbox("Blue 3", get_available_options(f"b3_{tier_key}"), key=f"b3_{tier_key}")
        b4 = st.selectbox("Blue 4", get_available_options(f"b4_{tier_key}"), key=f"b4_{tier_key}")
        b5 = st.selectbox("Blue 5", get_available_options(f"b5_{tier_key}"), key=f"b5_{tier_key}")

    with col2:
        st.subheader("🔴 Equipo RED")
        r1 = st.selectbox("Red 1", get_available_options(f"r1_{tier_key}"), key=f"r1_{tier_key}")
        r2 = st.selectbox("Red 2", get_available_options(f"r2_{tier_key}"), key=f"r2_{tier_key}")
        r3 = st.selectbox("Red 3", get_available_options(f"r3_{tier_key}"), key=f"r3_{tier_key}")
        r4 = st.selectbox("Red 4", get_available_options(f"r4_{tier_key}"), key=f"r4_{tier_key}")
        r5 = st.selectbox("Red 5", get_available_options(f"r5_{tier_key}"), key=f"r5_{tier_key}")

    blue_sel = normalize_selection([b1, b2, b3, b4, b5])
    red_sel  = normalize_selection([r1, r2, r3, r4, r5])

    st.divider()

    st.markdown(
        """
        - Para **evaluar un draft completo**, selecciona hasta 5 campeones por lado y pulsa **Calcular probabilidad**.  
        - Para **ver recomendaciones**, puedes dejar el draft vacío, aplicar bans o avanzar de forma secuencial.
        """
    )

    if st.button("🔍 Calcular probabilidad y recomendaciones", key=f"btn_{tier_key}"):
        
        # =========================================================================
        # 🟢 ESTADO A: El Draft está completamente limpio de picks (0 vs 0)
        # =========================================================================
        if len(blue_sel) == 0 and len(red_sel) == 0:
            st.markdown("## 📊 Resultado global del draft")
            st.info("Fase inicial del Draft (Picks: 0 vs 0). Revisa abajo las mejores aperturas del Meta.")
            
        # =========================================================================
        # 🔵 ESTADO B: El Draft ya comenzó (Ej: 1 vs 0, 2 vs 1, etc.)
        # =========================================================================
        else:
            try:
                # 1) PROBABILIDAD DE VICTORIA
                blue_ids = [normalize_champion(c) for c in blue_sel]
                red_ids = [normalize_champion(c) for c in red_sel]
                while len(blue_ids) < 5: blue_ids.append(-1)
                while len(red_ids) < 5: red_ids.append(-1)

                df_feats = build_features_for_draft(blue_ids, red_ids, tier=tier_key)
                p_blue = predict_blue_win_prob(df_feats, tier=tier_key)
                p_red = 1.0 - p_blue

                st.markdown("## 📊 Resultado global del draft")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Probabilidad de victoria BLUE", f"{p_blue*100:.1f}%")
                with c2:
                    st.metric("Probabilidad de victoria RED", f"{p_red*100:.1f}%")

                st.progress(p_blue)

                # 1.1) PANEL TÁCTICO DE LA CAPA GOLD
                st.markdown("### ⚙️ Resumen dimensional de la composición (Ventaja Relativa %)")
                gold_metrics = calculate_gold_deltas(blue_sel, red_sel)

                def fmt(val):
                    return f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"

                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Δ Control de Masas (CC)", fmt(gold_metrics['cc']))
                    st.metric("Δ Iniciación (Engage)", fmt(gold_metrics['engage']))
                    st.metric("Δ Hostigamiento (Poke)", fmt(gold_metrics['poke']))
                with colB:
                    st.metric("Δ Robustez (Durability)", fmt(gold_metrics['durability']))
                    st.metric("Δ Espaciado (Kiting)", fmt(gold_metrics['kiting']))
                    st.metric("Δ Utilidad / Mitigación", fmt(gold_metrics['utility']))
                with colC:
                    st.metric("Δ Presión Física (Phys Dmg)", fmt(gold_metrics['phys_dmg']))
                    st.metric("Δ Presión Mágica (Mag Dmg)", fmt(gold_metrics['mag_dmg']))
                    st.metric("Δ Daño Sostenido (DPS)", fmt(gold_metrics['dps']))

                resumen = coach_summary(gold_metrics)
                st.markdown(f"**Análisis del Coach:** {resumen}")

            except Exception as e:
                st.error(f"Error procesando la inferencia de la composición: {e}")
                st.stop()

        # =========================================================================
        # 🧠 2) RECOMENDACIONES DE PICKS (Enviando la lista 'ban_sel')
        # =========================================================================
        st.markdown("## 🧠 Recomendaciones de siguiente pick (Top-5)")

        if len(blue_sel) == 5 and len(red_sel) == 5:
            st.success("✨ ¡Draft finalizado! Ambos equipos han completado sus selecciones.")
        else:
            col_blue, col_red = st.columns(2)

            # 🔵 Columna de sugerencias para BLUE
            with col_blue:
                st.subheader("🔵 Sugerencias para BLUE")
                if len(blue_sel) == 5:
                    st.info("La composición de BLUE ya cuenta con sus 5 campeones.")
                else:
                    try:
                        recs_blue = recommend_for(blue_sel, red_sel, side="blue", top_k=5, tier=tier_key, bans=ban_sel)
                        if not recs_blue:
                            st.info("No hay candidatos viables disponibles.")
                        else:
                            for r in recs_blue:
                                st.markdown(
                                    f"**{r.champ_name}** \n"
                                    f"P(Blue Win): **{r.prob_blue_win*100:.1f}%** | Score: *{r.score:.3f}* \n"
                                    f"_{r.explanation}_"
                                )
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"Error calculando recomendaciones para BLUE: {e}")

            # 🔴 Columna de sugerencias para RED
            with col_red:
                st.subheader("🔴 Sugerencias para RED")
                if len(red_sel) == 5:
                    st.info("La composición de RED ya cuenta con sus 5 campeones.")
                else:
                    try:
                        recs_red = recommend_for(blue_sel, red_sel, side="red", top_k=5, tier=tier_key, bans=ban_sel)
                        if not recs_red:
                            st.info("No hay candidatos viables disponibles.")
                        else:
                            for r in recs_red:
                                st.markdown(
                                    f"**{r.champ_name}** \n"
                                    f"P(Red Win): **{r.prob_red_win*100:.1f}%** | Score: *{r.score:.3f}* \n"
                                    f"_{r.explanation}_"
                                )
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"Error calculando recomendaciones para RED: {e}")

# =====================================
# VISTA PRINCIPAL (TABS)
# =====================================

st.title("🎮 LoL Draft Recommender — Capa Gold & Multi-Tier")
st.markdown(
    """
    Simulador analítico basado en aprendizaje automático para drafts competitivos de League of Legends.  
    Selecciona la pestaña adecuada según el Elo de la sala para activar los pipelines correspondientes.
    """
)

st.divider()

# Separación estricta de las dos pestañas operacionales
tab_apex, tab_low = st.tabs(["🏆 Tier Apex (High Elo)", "🔰 Tier Low Elo"])

with tab_apex:
    st.markdown(
        """
        **Pipeline de Inferencia: APEX** *Datos entrenados con partidas de Master, Grandmaster y Challenger. Enfoque prioritario en sinergia fina, meta de alto rendimiento y counter-picks estrictos.*
        """
    )
    # Mandamos el identificador exacto de llave del modelo guardado ("apex")
    render_draft_interface(tier_key="apex")

with tab_low:
    st.markdown(
        """
        **Pipeline de Inferencia: LOW ELO** *Datos entrenados con ligas inferiores/intermedias. Maximiza el peso de los campeones con win rate bruto dominante en entornos con menor coordinación de equipo.*
        """
    )
    # Mandamos el identificador exacto de llave del modelo guardado ("lowtier")
    render_draft_interface(tier_key="lowtier")
