import streamlit as st
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================
# CONFIG Y IMPORTS
# =====================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

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
# UTILIDADES Y PROCESAMIENTO AVANZADO
# =====================================

def normalize_selection(selection_list):
    """Quita '(vacío)' y devuelve solo los nombres de campeones."""
    return [c for c in selection_list if c != "(vacío)"]

def get_champ_image_url(champ_name):
    """
    Obtiene la URL oficial del avatar del campeón desde Riot Data Dragon.
    Limpia signos de puntuación y mapea las excepciones de mayúsculas de Riot.
    """
    if champ_name == "(vacío)":
        return None
    
    row = RES.champs_df[RES.champs_df["name"] == champ_name]
    if not row.empty and "apiname" in row.columns and pd.notna(row["apiname"]).any():
        api_name = str(row["apiname"].values[0]).strip()
    else:
        api_name = champ_name.replace(" ", "").replace("'", "").replace(".", "").replace(",", "").replace("_", "")
        
        # Diccionario de excepciones corregido con MasterYi incluido
        mapping = {
            "nunu&willump": "Nunu",
            "aurelionsol": "AurelionSol",
            "xinzhao": "XinZhao",
            "masteryi": "MasterYi",
            "belveth": "Belveth",
            "drmundo": "DrMundo",
            "wukong": "MonkeyKing",
            "chogath": "Chogath",
            "khazix": "Khazix",
            "reksai": "RekSai",
            "renataglasc": "Renata",
            "kaisa": "KaiSa",
            "leblanc": "LeBlanc",
            "ksante": "KSante",
            "twistedfate": "TwistedFate",
            "fiddlesticks": "FiddleSticks"
        }
        
        key = api_name.lower()
        if key in mapping:
            api_name = mapping[key]
        else:
            api_name = api_name.capitalize()

    return f"https://ddragon.leagueoflegends.com/cdn/15.24.1/img/champion/{api_name}.png"


def calculate_gold_metrics(blue_sel, red_sel):
    """
    Calcula los valores absolutos compartidos (para el gráfico de radar)
    y las ventajas relativas en porcentaje (para los paneles de métricas).
    """
    blue_ids = [normalize_champion(c) for c in blue_sel]
    red_ids = [normalize_champion(c) for c in red_sel]
    
    b_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in blue_ids], axis=0)
    r_v = np.sum([RES.champ_vectors.get(c, np.zeros(9)) for c in red_ids], axis=0)
    
    total_v = b_v + r_v
    
    # % de ventaja relativa para st.metric
    delta_pct = np.where(total_v > 0, ((b_v - r_v) / total_v) * 100, 0.0)
    
    # Distribución de poder (0-100) para el gráfico de radar
    b_share = np.where(total_v > 0, (b_v / total_v) * 100, 50.0)
    r_share = np.where(total_v > 0, (r_v / total_v) * 100, 50.0)
    
    categories = ["Físico", "Mágico", "DPS", "Tanque", "CC", "Poke", "Engage", "Utilidad", "Kiting"]
    
    metrics = {
        "phys_dmg": delta_pct[0], "mag_dmg": delta_pct[1], "dps": delta_pct[2],
        "durability": delta_pct[3], "cc": delta_pct[4], "poke": delta_pct[5],
        "engage": delta_pct[6], "utility": delta_pct[7], "kiting": delta_pct[8]
    }
    
    return metrics, b_share, r_share, categories


def plot_radar_chart(b_share, r_share, categories):
    """Genera un gráfico de radar interactivo comparando ambos equipos."""
    categories_loop = categories + [categories[0]]
    b_loop = list(b_share) + [b_share[0]]
    r_loop = list(r_share) + [r_share[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=b_loop, theta=categories_loop, fill='toself', name='BLUE Team',
        line_color='#4A90E2', fillcolor='rgba(74, 144, 226, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_loop, theta=categories_loop, fill='toself', name='RED Team',
        line_color='#E03A3E', fillcolor='rgba(224, 58, 62, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def coach_summary(m: dict) -> str:
    """Diagnóstico estratégico basado en las ventajas porcentuales."""
    lines = []
    if m["cc"] > 10.0: lines.append("BLUE tiene **mayor densidad de CC (+)**, ideal para asegurar cazadas.")
    elif m["cc"] < -10.0: lines.append("RED cuenta con **ventaja notable en CC**, BLUE debe cuidar sus líneas de entrada.")
    if m["engage"] > 10.0: lines.append("BLUE cuenta con **mejor engage directo**, manteniendo la iniciativa.")
    elif m["engage"] < -10.0: lines.append("BLUE está en **desventaja de iniciación**; juegue al counter-engage.")
    if m["durability"] > 12.0: lines.append("BLUE posee una **frontline más sólida**, óptima para peleas extendidas.")
    elif m["durability"] < -12.0: lines.append("RED es **más robusto**, BLUE debe evitar choques frontales directos.")
    if m["poke"] > 10.0: lines.append("BLUE destaca en **Poke**, busque desgastar antes de los objetivos.")
    if m["kiting"] > 10.0: lines.append("BLUE tiene **mejor Kiting**, excelente para estirar la pelea hacia atrás.")
    if m["phys_dmg"] > 20.0 and m["mag_dmg"] < -20.0: lines.append("BLUE está **sesgado a daño físico**, RED armará armadura.")
    elif m["mag_dmg"] > 20.0 and m["phys_dmg"] < -20.0: lines.append("BLUE tiene **exceso de daño mágico**, RED acumulará resistencia.")

    return " ".join(lines) if lines else "Las composiciones se encuentran altamente equilibradas en sus ejes tácticos."


# =====================================
# INTERFAZ MODULAR DE DRAFT
# =====================================
def callback_limpiar_draft(tier_key):
    """
    Función Callback que ejecuta la limpieza de los estados antes 
    de que Streamlit renderice los widgets de la pantalla.
    """
    for i in range(1, 6):
        st.session_state[f"b{i}_{tier_key}"] = "(vacío)"
        st.session_state[f"r{i}_{tier_key}"] = "(vacío)"
        st.session_state[f"bb{i}_{tier_key}"] = "(vacío)"
        st.session_state[f"rb{i}_{tier_key}"] = "(vacío)"
        
def render_draft_interface(tier_key):
    champ_list = sorted(RES.champs_df["name"].tolist())

    # Filtrado dinámico estricto anti-duplicados (Picks y Bans cruzados)
    def get_available_options(current_key):
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
                if val != "(vacío)": selected_elsewhere.add(val)
        return ["(vacío)"] + [c for c in champ_list if c not in selected_elsewhere]

    # ==========================================
    # 🚫 INTERFAZ DE BANS (Fase de Bloqueos)
    # ==========================================
    with st.expander("🚫 Fase de Bloqueos (Bans)", expanded=False):
        b_col, r_col = st.columns(2)
        with b_col:
            st.markdown(":blue[**Bans Equipo BLUE**]")
            bb_cols = st.columns(5)
            bb1 = bb_cols[0].selectbox("B1", get_available_options(f"bb1_{tier_key}"), key=f"bb1_{tier_key}", label_visibility="collapsed")
            bb2 = bb_cols[1].selectbox("B2", get_available_options(f"bb2_{tier_key}"), key=f"bb2_{tier_key}", label_visibility="collapsed")
            bb3 = bb_cols[2].selectbox("B3", get_available_options(f"bb3_{tier_key}"), key=f"bb3_{tier_key}", label_visibility="collapsed")
            bb4 = bb_cols[3].selectbox("B4", get_available_options(f"bb4_{tier_key}"), key=f"bb4_{tier_key}", label_visibility="collapsed")
            bb5 = bb_cols[4].selectbox("B5", get_available_options(f"bb5_{tier_key}"), key=f"bb5_{tier_key}", label_visibility="collapsed")
        with r_col:
            st.markdown(":red[**Bans Equipo RED**]")
            rb_cols = st.columns(5)
            rb1 = rb_cols[0].selectbox("R1", get_available_options(f"rb1_{tier_key}"), key=f"rb1_{tier_key}", label_visibility="collapsed")
            rb2 = rb_cols[1].selectbox("R2", get_available_options(f"rb2_{tier_key}"), key=f"rb2_{tier_key}", label_visibility="collapsed")
            rb3 = rb_cols[2].selectbox("R3", get_available_options(f"rb3_{tier_key}"), key=f"rb3_{tier_key}", label_visibility="collapsed")
            rb4 = rb_cols[3].selectbox("R4", get_available_options(f"rb4_{tier_key}"), key=f"rb4_{tier_key}", label_visibility="collapsed")
            rb5 = rb_cols[4].selectbox("R5", get_available_options(f"rb5_{tier_key}"), key=f"rb5_{tier_key}", label_visibility="collapsed")

    ban_sel = normalize_selection([bb1, bb2, bb3, bb4, bb5, rb1, rb2, rb3, rb4, rb5])

    # ==========================================
    # ⚔️ INTERFAZ DE PICKS (Sin etiquetas de nombre)
    # ==========================================
    col1, col2 = st.columns(2)
    blue_picks = []
    red_picks = []

    with col1:
        st.subheader("🔵 Equipo BLUE")
        for i in range(1, 6):
            img_col, select_col = st.columns([1, 5])
            current_key = f"b{i}_{tier_key}"
            # Se colapsa el label para dejar el selector completamente libre de texto superior
            chosen = select_col.selectbox(f"Blue {i}", get_available_options(current_key), key=current_key, label_visibility="collapsed")
            blue_picks.append(chosen)
            img_url = get_champ_image_url(chosen)
            if img_url: 
                img_col.image(img_url, width=54)

    with col2:
        st.subheader("🔴 Equipo RED")
        for i in range(1, 6):
            img_col, select_col = st.columns([1, 5])
            current_key = f"r{i}_{tier_key}"
            # Se colapsa el label para dejar el selector completamente libre de texto superior
            chosen = select_col.selectbox(f"Red {i}", get_available_options(current_key), key=current_key, label_visibility="collapsed")
            red_picks.append(chosen)
            img_url = get_champ_image_url(chosen)
            if img_url: 
                img_col.image(img_url, width=54)

    blue_sel = normalize_selection(blue_picks)
    red_sel  = normalize_selection(red_picks)

    st.divider()

    # Botones de Acción principales colocados lado a lado
    btn_col1, btn_col2, _ = st.columns([2, 1, 4])
    calc_pressed = btn_col1.button("🔍 Calcular probabilidad y recomendaciones", key=f"btn_{tier_key}")
    
    # Botón de Reinicio Rápido
    btn_col2.button(
            "🧹 Limpiar Draft", 
            key=f"clear_{tier_key}", 
            on_click=callback_limpiar_draft, 
            args=(tier_key,)
        )

    if calc_pressed:
        if len(blue_sel) == 0 and len(red_sel) == 0:
            st.markdown("## 📊 Resultado global del draft")
            st.info("Fase inicial del Draft (0 vs 0). Abajo verás las mejores aperturas estructurales.")
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
                with c1: st.metric("Probabilidad de victoria BLUE", f"{p_blue*100:.1f}%")
                with c2: st.metric("Probabilidad de victoria RED", f"{p_red*100:.1f}%")
                st.progress(p_blue)

                # Panel Táctico Combinado (Gráfico + Métricas)
                st.markdown("### ⚙️ Resumen dimensional de la composición (Ventaja Relativa %)")
                gold_metrics, b_share, r_share, categories = calculate_gold_metrics(blue_sel, red_sel)
                
                chart_col, metrics_col = st.columns([1.3, 1.7])
                
                with chart_col:
                    radar_fig = plot_radar_chart(b_share, r_share, categories)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with metrics_col:
                    def fmt(val): return f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"
                    mA, mB, mC = st.columns(3)
                    with mA:
                        st.metric("Δ Control de Masas", fmt(gold_metrics['cc']))
                        st.metric("Δ Iniciación (Engage)", fmt(gold_metrics['engage']))
                        st.metric("Δ Hostigamiento (Poke)", fmt(gold_metrics['poke']))
                    with mB:
                        st.metric("Δ Robustez (Tanque)", fmt(gold_metrics['durability']))
                        st.metric("Δ Espaciado (Kiting)", fmt(gold_metrics['kiting']))
                        st.metric("Δ Utilidad / Mitigación", fmt(gold_metrics['utility']))
                    with mC:
                        st.metric("Δ Presión Física", fmt(gold_metrics['phys_dmg']))
                        st.metric("Δ Presión Mágica", fmt(gold_metrics['mag_dmg']))
                        st.metric("Δ Daño Sostenido", fmt(gold_metrics['dps']))

                st.markdown(f"**Análisis del Coach:** {coach_summary(gold_metrics)}")

            except Exception as e:
                st.error(f"Error procesando la inferencia: {e}")
                st.stop()

        # ==========================================
        # 🧠 RECOMENDACIONES TOP-5 (Con Retratos Integrados)
        # ==========================================
        st.markdown("## 🧠 Recomendaciones de siguiente pick (Top-5)")

        if len(blue_sel) == 5 and len(red_sel) == 5:
            st.success("✨ ¡Draft finalizado! Ambos equipos han completado sus selecciones.")
        else:
            col_blue, col_red = st.columns(2)

            with col_blue:
                st.subheader("🔵 Sugerencias para BLUE")
                if len(blue_sel) == 5:
                    st.info("BLUE ya completó sus 5 picks.")
                else:
                    try:
                        recs_blue = recommend_for(blue_sel, red_sel, side="blue", top_k=5, tier=tier_key, bans=ban_sel)
                        for r in recs_blue:
                            img_c, text_c = st.columns([1, 5])
                            url = get_champ_image_url(r.champ_name)
                            if url: img_c.image(url, width=60)
                            
                            # Formato limpio: El porcentaje al lado del nombre
                            text_c.markdown(
                                f"**{r.champ_name}** — Probabilidad: **{r.prob_blue_win*100:.1f}%**\n"
                                f"Score Táctico: *{r.score:.3f}*\n"
                                f"_{r.explanation}_"
                            )
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Error en sugerencias de BLUE: {e}")

            with col_red:
                st.subheader("🔴 Sugerencias para RED")
                if len(red_sel) == 5:
                    st.info("RED ya completó sus 5 picks.")
                else:
                    try:
                        recs_red = recommend_for(blue_sel, red_sel, side="red", top_k=5, tier=tier_key, bans=ban_sel)
                        for r in recs_red:
                            img_c, text_c = st.columns([1, 5])
                            url = get_champ_image_url(r.champ_name)
                            if url: img_c.image(url, width=60)
                            
                            # Formato limpio: El porcentaje al lado del nombre
                            text_c.markdown(
                                f"**{r.champ_name}** — Probabilidad: **{r.prob_red_win*100:.1f}%**\n"
                                f"Score Táctico: *{r.score:.3f}*\n"
                                f"_{r.explanation}_"
                            )
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Error en sugerencias de RED: {e}")


# =====================================
# VISTA PRINCIPAL (TABS)
# =====================================

st.title("🎮 LoL Draft Recommender — Versión Profesional")
st.markdown(
    """
    Simulador estratégico avanzado con análisis multidimensional de la Capa Gold y motores de inferencia predictiva segmentados.
    """
)

st.divider()

tab_apex, tab_low = st.tabs(["🏆 Tier Apex (High Elo)", "🔰 Tier Low Elo"])

with tab_apex:
    st.markdown("**Pipeline de Inferencia: APEX** (Partidas de Master a Challenger). Foco en Meta de rendimiento y counter-picks.")
    render_draft_interface(tier_key="apex")

with tab_low:
    st.markdown("**Pipeline de Inferencia: LOW ELO** (Partidas de Iron hasta Emerald). Foco en win rates brutos y confort.")
    render_draft_interface(tier_key="lowtier")
