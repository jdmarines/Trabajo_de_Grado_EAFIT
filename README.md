# Sistema de Recomendación Secuencial en Entornos de Baja Separabilidad Estadística: Aplicación al Draft de League of Legends

Este repositorio contiene el código fuente, los Notebooks experimentales, los artefactos de modelado y la interfaz de producción desarrollados como Trabajo de Grado para la **Maestría en Ciencias de los Datos y la Analítica** de la **Universidad EAFIT** (2026)

* **Autor:** Juan David Marín Escobar 
* **Asesor:** Juan David Martínez Vargas 
* **Institución:** Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería 
* **Ubicación y Año:** Medellín, 2026 

---

## 📝 Resumen Ejecutivo del Proyecto

Esta investigación aborda el complejo problema de la toma de decisiones secuenciales en escenarios tabulares discretos caracterizados por una relación señal-ruido extremadamente baja. Tomando como caso de estudio la fase de selección de personajes (*draft*) en el videojuego *League of Legends*, el proyecto desafía la expectativa convencional de construir clasificadores con métricas de exactitud elevadas, demostrando en su lugar que es científicamente viable extraer señales estratégicas marginales y ventajas probabilísticas útiles dentro de un dominio deliberadamente balanceado y regularizado de forma externa.

El entorno experimental está condicionado por el *Champion Balance Framework* de Riot Games, un mecanismo algorítmico de optimización continua que interviene quincenalmente la plantilla de personajes para forzar de manera artificial que sus tasas de éxito basales converjan simétricamente alrededor del equilibrio del 50%. Esta contención de la varianza estructural genera un escenario de bajísima separabilidad estadística

Para mitigar esta restricción, se diseñó e implementó un pipeline de ingeniería de características denominado **Capa Gold**, el cual ejecuta una reducción semántica de la dimensionalidad. Al proyectar identidades categóricas dispersas hacia métricas macroestructurales continuas y ratios relacionales basados en los principios no cooperativos de la Teoría de Juegos, el sistema logra extraer señal estadística explotable. Los resultados del torneo de modelos evidencian una marcada asimetría analítica según el nivel de destreza (*skill tier*) de los jugadores, consolidando un óptimo global de $AUC = 0.566589$ mediante un ensamble XGBoost en escenarios de alta entropía operativa humana.

---

## Link del sistema de recomendacion en stramlit : https://loldraft.streamlit.app

---
## 📂 Arquitectura  del Repositorio

```text
Trabajo_de_Grado_EAFIT/
│
├── Notebooks/                          # Cuadernos experimentales de desarrollo 
│   ├── Descarga_de_partidas.ipynb      # Pipeline de extracción y consumo de la API de Riot Games 
│   ├── silver_layer.ipynb              # Modelado paramétrico temporal a nivel crítico 13 
│   ├── gold_layer.ipynb                # Orquestación de características macroestructurales continuas 
│   └── Entrenamiento_de_modelos.ipynb  # Benchmarking, análisis de ablación y Grid Search (k=3) 
│
├── data/                               # Repositorio de almacenamiento de datos y matrices 
│   ├── notebooks/                      # Subcarpeta interna con sets intermedios y consolidados 
│   │   ├── matches_high_elo.csv        # Registro de partidas crudas del segmento Apex Tier 
│   │   ├── matches_low_elo.csv         # Registro de partidas crudas del segmento LowTier 
│   │   ├── players_apex_seed.json      # Semilla de invocadores de alto rango para la API 
│   │   ├── capa_silver_final.csv       # Atributos planos transformados a nivel 13 
│   │   ├── capa_gold_final.xlsx        # Matriz relacional macroestructural densa de modelado 
│   │   └── winrate.xlsx                # Tasas de victoria históricas basales del parche 26.8 
│   └── champs_metadata.csv             # Metadatos estáticos base de los 172 campeones del juego 
│
├── models/                             # Serialización de los estimadores óptimos (Joblib) 
│   ├── model_apex.joblib               # Clasificador optimizado para el segmento Apex Tier 
│   └── model_low_elo.joblib            # Clasificador optimizado para el segmento LowTier 
│
├── src/                                # Código fuente modular auxiliar 
│   └── recommender.py                  # Algoritmo de ordenamiento lógico y beneficio marginal 
│
├── README.md                           # Documentación técnica principal (este archivo) 
├── app.py                              # Interfaz web interactiva de producción en Streamlit 
└── requirements.txt                    # Manifiesto de dependencias y librerías del proyecto

