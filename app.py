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
    Calcula de forma exacta los diferenciales de la Capa Gold de 9 dimensiones
    para el panel visual de la UI, independiente de la etapa del modelo.
    """
    blue_ids = [normalize_champion(c) for c in blue_sel]
    red_ids = [normalize_champion(c) for c in red_sel]
    
    # Sumar vectores de la capa Gold por equipo
    b_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in blue_ids], axis=0)
    r_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in red_ids], axis=0)
    
    deltas = b_v - r_v
    
    return {
        "phys_dmg": deltas[0],
        "mag_dmg": deltas[1],
        "dps": deltas[2],
        "durability": deltas[3],
        "cc": deltas[4],
        "poke": deltas[5],
        "engage": deltas[6],
        "utility": deltas[7],
        "kiting": deltas[8]
    }

def coach_summary(m: dict) -> str:
    """
    Genera un diagnóstico estratégico basado en las 9 dimensiones Gold
    interpretado siempre a favor de las ventajas/desventajas de BLUE.
    """
    lines = []

    # Control de Masas (CC)
    if m["cc"] > 1.0:
        lines.append("BLUE tiene **más CC total**, ideal para asegurar cazadas y controlar peleas grupales.")
    elif m["cc"] < -1.0:
        lines.append("RED cuenta con **mayor densidad de CC**, BLUE debe cuidar los ángulos y limpiar marcas de control.")

    # Iniciación (Engage)
    if m["engage"] > 1.0:
        lines.append("La composición BLUE cuenta con **mejor engage directo**, tiene la iniciativa para forzar peleas.")
    elif m["engage"] < -1.0:
        lines.append("BLUE tiene **menos iniciación**; conviene jugar al counter-engage, desenganche o transiciones lentas.")

    # Durabilidad (Frontline)
    if m["durability"] > 1.5:
        lines.append("BLUE posee una **frontline más sólida/resistente**, óptima para combates de desgaste extendidos.")
    elif m["durability"] < -1.5:
        lines.append("RED es **significativamente más robusto**, BLUE debe evitar los duelos frontales prolongados.")

    # Desgaste y Espaciado (Poke & Kiting)
    if m["poke"] > 1.5:
        lines.append("BLUE destaca en **capacidad de Poke**, busque debilitar al rival antes de pelear los objetivos neutrales.")
    if m["kiting"] > 1.0:
        lines.append("BLUE tiene **mejor perfil de Kiting**, lo que le facilita castigar al rival mientras retrocede.")

    # Balance de Tipos de Daño
    if m["phys_dmg"] > m["mag_dmg"] and m["phys_dmg"] > 250:
        lines.append("La composición de BLUE está **sesgada hacia el daño físico**, alertando la acumulación de armadura en RED.")
    elif m["mag_dmg"] > m["phys_dmg"] and m["mag_dmg"] > 250:
        lines.append("BLUE ejerce **fuerte presión de daño mágico**, obligando a RED a priorizar resistencia mágica temprana.")

    if not lines:
        return "Las composiciones se encuentran equilibradas en sus ejes tácticos principales."

    return " ".join(lines)


# =====================================
# INTERFAZ MODULAR DE DRAFT
# =====================================

def render_draft_interface(tier_key):
    """
    Renderiza el simulador de draft para un tier específico ('apex' o 'lowtier').
    """
    # Catálogo unificado obtenido del dataframe cargado en el backend
    champ_list = sorted(RES.champs_df["name"].tolist())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Equipo BLUE")
        b1 = st.selectbox("Blue 1", ["(vacío)"] + champ_list, key=f"b1_{tier_key}")
        b2 = st.selectbox("Blue 2", ["(vacío)"] + champ_list, key=f"b2_{tier_key}")
        b3 = st.selectbox("Blue 3", ["(vacío)"] + champ_list, key=f"b3_{tier_key}")
        b4 = st.selectbox("Blue 4", ["(vacío)"] + champ_list, key=f"b4_{tier_key}")
        b5 = st.selectbox("Blue 5", ["(vacío)"] + champ_list, key=f"b5_{tier_key}")

    with col2:
        st.subheader("🔴 Equipo RED")
        r1 = st.selectbox("Red 1", ["(vacío)"] + champ_list, key=f"r1_{tier_key}")
        r2 = st.selectbox("Red 2", ["(vacío)"] + champ_list, key=f"r2_{tier_key}")
        r3 = st.selectbox("Red 3", ["(vacío)"] + champ_list, key=f"r3_{tier_key}")
        r4 = st.selectbox("Red 4", ["(vacío)"] + champ_list, key=f"r4_{tier_key}")
        r5 = st.selectbox("Red 5", ["(vacío)"] + champ_list, key=f"r5_{tier_key}")

    blue_sel = normalize_selection([b1, b2, b3, b4, b5])
    red_sel  = normalize_selection([r1, r2, r3, r4, r5])

    st.divider()

    st.markdown(
        """
        - Para **evaluar un draft completo**, selecciona hasta 5 campeones por lado y pulsa **Calcular probabilidad**.  
        - Para **ver recomendaciones top-5**, usa **máximo 4 campeones por lado** (simulando un draft en progreso).
        """
    )

    if st.button("🔍 Calcular probabilidad y recomendaciones", key=f"btn_{tier_key}"):
        if len(blue_sel) == 0 or len(red_sel) == 0:
            st.error("Debes seleccionar al menos un campeón en cada equipo.")
        else:
            try:
                # ==========================
                # 1) PROBABILIDAD DE VICTORIA
                # ==========================
                # Transformamos los nombres a IDs de campeones y rellenamos con -1 como pide el modelo
                blue_ids = [normalize_champion(c) for c in blue_sel]
                red_ids = [normalize_champion(c) for c in red_sel]
                while len(blue_ids) < 5: blue_ids.append(-1)
                while len(red_ids) < 5: red_ids.append(-1)

                # Inferencia con el modelo correspondiente del tier asignado
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

                # ==========================
                # 1.1 PANEL TÁCTICO DE LA CAPA GOLD
                # ==========================
                st.markdown("### ⚙️ Resumen dimensional de la composición (BLUE vs RED)")
                
                # Extracción manual de deltas dimensionales puras para estabilidad de UI
                gold_metrics = calculate_gold_deltas(blue_sel, red_sel)

                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Δ Control de Masas (CC)", f"{gold_metrics['cc']:.2f}")
                    st.metric("Δ Iniciación (Engage)", f"{gold_metrics['engage']:.2f}")
                    st.metric("Δ Hostigamiento (Poke)", f"{gold_metrics['poke']:.2f}")

                with colB:
                    st.metric("Δ Robustez (Durability)", f"{gold_metrics['durability']:.2f}")
                    st.metric("Δ Espaciado (Kiting)", f"{gold_metrics['kiting']:.2f}")
                    st.metric("Δ Utilidad / Mitigación", f"{gold_metrics['utility']:.2f}")

                with colC:
                    st.metric("Δ Presión Física (Phys Dmg)", f"{gold_metrics['phys_dmg']:.1f}")
                    st.metric("Δ Presión Mágica (Mag Dmg)", f"{gold_metrics['mag_dmg']:.1f}")
                    st.metric("Δ Daño Sostenido (DPS)", f"{gold_metrics['dps']:.1f}")

                # ==========================
                # 1.2 COMENTARIO TIPO COACH
                # ==========================
                resumen = coach_summary(gold_metrics)
                st.markdown(f"**Análisis del Coach:** {resumen}")

            except Exception as e:
                st.error(f"Error procesando la inferencia de la composición: {e}")

# ==========================
            # 2) RECOMENDACIONES TOP-5
            # ==========================
            st.markdown("## 🧠 Recomendaciones de siguiente pick (Top-5)")

            # Solo asumimos que el draft cerró por completo si ambos lados están llenos
            if len(blue_sel) == 5 and len(red_sel) == 5:
                st.success("✨ ¡Draft finalizado! Ambos equipos han completado sus selecciones.")
            else:
                col_blue, col_red = st.columns(2)

                # 🔵 Recomendaciones para BLUE
                with col_blue:
                    st.subheader("🔵 Sugerencias para BLUE")
                    if len(blue_sel) == 5:
                        st.info("La composición de BLUE ya cuenta con sus 5 campeones.")
                    else:
                        try:
                            recs_blue = recommend_for(blue_sel, red_sel, side="blue", top_k=5, tier=tier_key)
                            if not recs_blue:
                                st.info("No hay candidatos viables disponibles.")
                            else:
                                for r in recs_blue:
                                    st.markdown(
                                        f"**{r.champ_name}** \n"
                                        f"P(Blue Win): **{r.prob_blue_win*100:.1f}%** | Score Táctico: *{r.score:.3f}* \n"
                                        f"_{r.explanation}_"
                                    )
                                    st.markdown("---")
                        except Exception as e:
                            st.error(f"Error calculando recomendaciones para BLUE: {e}")

                # 🔴 Recomendaciones para RED
                with col_red:
                    st.subheader("🔴 Sugerencias para RED")
                    if len(red_sel) == 5:
                        st.info("La composición de RED ya cuenta con sus 5 campeones.")
                    else:
                        try:
                            recs_red = recommend_for(blue_sel, red_sel, side="red", top_k=5, tier=tier_key)
                            if not recs_red:
                                st.info("No hay candidatos viables disponibles.")
                            else:
                                for r in recs_red:
                                    st.markdown(
                                        f"**{r.champ_name}** \n"
                                        f"P(Red Win): **{r.prob_red_win*100:.1f}%** | Score Táctico: *{r.score:.3f}* \n"
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
