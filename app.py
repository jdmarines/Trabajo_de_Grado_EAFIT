import streamlit as st
from pathlib import Path
import sys

# Configuración de rutas
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from recommender import (
    RES_APEX,          # Recursos cargados para Apex
    RES_LOW,           # Recursos cargados para Low Elo
    build_features_for_draft,
    predict_blue_win_prob,
    recommend_for,
)

st.set_page_config(
    page_title="LoL Draft Recommender",
    page_icon="🎮",
    layout="wide",
)

# ... (Tus funciones auxiliares como normalize_selection y coach_summary se quedan igual) ...

def render_draft_interface(tier_name, resources, tier_key):
    """
    Función modular para renderizar la interfaz de draft.
    Evita duplicar el código de la UI para cada pestaña.
    """
    # Aquí va la lista de campeones específica de ese tier si varía
    champs_df = resources.champs_df
    champ_list = sorted(champs_df["name"].tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔵 Equipo BLUE")
        b1 = st.selectbox("Blue 1", ["(vacío)"] + champ_list, key=f"b1_{tier_key}")
        # ... b2, b3, b4, b5 usando la misma lógica de key única ...
        
    with col2:
        st.subheader("🔴 Equipo RED")
        r1 = st.selectbox("Red 1", ["(vacío)"] + champ_list, key=f"r1_{tier_key}")
        # ... r2, r3, r4, r5 ...

    blue_sel = normalize_selection([b1, b2, b3, b4, b5]) # (Y los demás picks)
    red_sel  = normalize_selection([r1, r2, r3, r4, r5])

    st.divider()
    
    if st.button("🔍 Calcular probabilidad y recomendaciones", key=f"btn_{tier_key}"):
        if len(blue_sel) == 0 or len(red_sel) == 0:
            st.error("Debes seleccionar al menos un campeón en cada equipo.")
            return

        # 1) Predicción pasando los recursos o el identificador del tier
        # Aquí ajustas según tu nueva metodología si los métodos cambian por tier
        feats = build_features_for_draft(blue_sel, red_sel, tier=tier_key)
        p_blue = predict_blue_win_prob(feats, tier=tier_key)
        p_red = 1.0 - p_blue

        # ... (Renderizar métricas, barra de progreso y coach_summary igual que antes) ...

        # 2) Recomendaciones
        # Pasas el parámetro tier para que el motor busque en el modelo correcto
        recs_blue = recommend_for(blue_sel, red_sel, side="blue", top_k=5, tier=tier_key)
        # ... Renderizar sugerencias ...


# =====================================
# FLUJO PRINCIPAL DE LA UI
# =====================================
st.title("🎮 LoL Draft Recommender Pro")

# Creación de las dos pestañas principales
tab_apex, tab_low = st.tabs(["🏆 Tier Apex (High Elo)", "🔰 Tier Low Elo"])

with tab_apex:
    st.markdown("### Modelado basado en Master, Grandmaster y Challenger")
    render_draft_interface("Apex", RES_APEX, tier_key="apex")

with tab_low:
    st.markdown("### Modelado basado en Iron, Bronze, Silver y Gold")
    render_draft_interface("Low Elo", RES_LOW, tier_key="low")
