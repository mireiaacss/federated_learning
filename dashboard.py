"""
Streamlit dashboard for the Federated Learning results.
Ejecuta primero federated_learning.py y luego lanza esto con: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# configuración principal de la página de streamlit
st.set_page_config(
    page_title="FL Model Performance | Bank Loan Risk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# diccionario de colores para mantener consistencia visual en todo el dashboard
COLORS = {
    "federated":   "#3B82F6",
    "centralized": "#F59E0B",
    "positive":    "#10B981",
    "negative":    "#EF4444",
    "bg":          "#0F172A",
    "card":        "#1E293B",
}

# plantilla base para los gráficos de plotly (fondo transparente y texto claro)
base_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F8FAFC", size=13),
)

# función para cargar los datos de los csv con cuidado por si no hemos entrenado el modelo aún
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

# cargamos todos los archivos que escupió nuestro modelo
round_log   = load_csv("round_log.csv")
results_df  = load_csv("results_summary.csv")
bank_df     = load_csv("bank_metrics.csv")
cm_df       = load_csv("confusion_matrices.csv")
eda_df      = load_csv("eda_analysis.csv")

# comprobamos que están todos para no petar la app
data_ready  = all(d is not None for d in [round_log, results_df, bank_df, cm_df, eda_df])

# metemos un poco de css a la fuerza para que el modo oscuro se vea perfecto y el texto resalte
st.markdown("""
<style>
    /* Forzamos fondo oscuro para que el texto blanco se lea sí o sí */
    .stApp { background-color: #0F172A !important; }
    .stApp, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #E2E8F0 !important; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; font-weight: 600 !important; }
    
    section[data-testid="stSidebar"] { background-color: #0B1120 !important; border-right: 1px solid #1E293B; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li { color: #E2E8F0 !important; font-size: 0.95rem; }
    section[data-testid="stSidebar"] strong { color: #60A5FA !important; font-weight: 600; }
    section[data-testid="stSidebar"] hr { border-color: #1E293B !important; margin: 1.5rem 0; }
    
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; margin: 4px 0; color: #FFFFFF; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .metric-delta { font-size: 0.85rem; margin-top: 4px; font-weight: 500; }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 16px;
        margin-top: 24px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }
    
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #3B82F6; }
</style>
""", unsafe_allow_html=True)

# barra lateral con los metadatos del modelo (tipo ficha técnica)
with st.sidebar:
    st.markdown("### Model Architecture")
    st.markdown("- **Base Estimator:** Logistic Regression")
    st.markdown("- **Algorithm:** Federated Averaging (FedAvg)")
    st.markdown("- **Nodes:** 3 Independent Banks")
    
    st.markdown("---")
    
    st.markdown("### Dataset Specs")
    st.markdown("- **Domain:** Financial Services (Credit Risk)")
    st.markdown("- **Target:** Loan Eligibility (Binary)")
    st.markdown("- **Class Imbalance:** 88% / 12%")
    st.markdown("- **Preprocessing:** SMOTE (Minority Oversampling)")

if not data_ready:
    st.error("Faltan datos. Corre primero el script de python (federated_learning.py) para generar los CSVs.")
    st.stop()

# sacamos las filas principales para pintar las métricas de un vistazo
fed_row = results_df[results_df["label"].str.contains("federated", case=False)].iloc[0]
cen_row = results_df[results_df["label"].str.contains("centralized", case=False)].iloc[0]

# cabecera principal del dashboard
st.markdown("## Federated Learning Evaluation Dashboard")
st.markdown("##### Comparative performance between distributed and centralized banking models")

# definimos las pestañas de navegación
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Global Metrics",
    "Convergence",
    "Node Analytics",
    "Evaluation Matrices",
    "Dataset Analysis"
])

# pestaña 1: kpis principales y comparativa global
with tab1:
    st.markdown('<div class="section-title">Primary Evaluation Metrics</div>', unsafe_allow_html=True)

    metrics = [
        ("F1 Score (Macro)", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("ROC-AUC", "auc"),
    ]

    # bucle para crear las tarjetas de métricas emparejadas de dos en dos
    for i in range(0, len(metrics), 2):
        cols = st.columns(2)
        for j, (name, key) in enumerate(metrics[i:i+2]):
            with cols[j]:
                fed_val = fed_row[key]
                cen_val = cen_row[key]
                delta   = fed_val - cen_val
                # verde si es igual o pierde muy poco, rojo si el modelo federado es mucho peor
                delta_color = "#10B981" if delta >= -0.02 else "#EF4444"
                delta_sign  = "+" if delta >= 0 else ""
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div style="display:flex; gap:32px; align-items:baseline;">
                        <div>
                            <div style="font-size:0.75rem;color:#60A5FA;margin-bottom:2px;font-weight:700;">FEDERATED</div>
                            <div class="metric-value">{fed_val:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem;color:#FBBF24;margin-bottom:2px;font-weight:700;">BASELINE (CENTRAL)</div>
                            <div class="metric-value">{cen_val:.3f}</div>
                        </div>
                    </div>
                    <div class="metric-delta" style="color:{delta_color};">
                        Δ {delta_sign}{delta:.4f} variance
                    </div>
                </div>
                """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Metric Distribution</div>', unsafe_allow_html=True)

        metric_names = ["F1", "Precision", "Recall", "ROC-AUC", "Accuracy"]
        metric_keys  = ["f1", "precision", "recall", "auc", "accuracy"]

        # pintamos un gráfico de barras comparando cara a cara los dos modelos
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Federated",
            x=metric_names,
            y=[fed_row[k] for k in metric_keys],
            marker_color=COLORS["federated"],
            text=[f"{fed_row[k]:.3f}" for k in metric_keys],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12)
        ))
        fig.add_trace(go.Bar(
            name="Centralized Baseline",
            x=metric_names,
            y=[cen_row[k] for k in metric_keys],
            marker_color=COLORS["centralized"],
            text=[f"{cen_row[k]:.3f}" for k in metric_keys],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12)
        ))
        
        layout_bar = base_layout.copy()
        layout_bar.update(
            barmode="group",
            yaxis=dict(range=[0, 1.15], gridcolor="#334155", showgrid=True),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=360,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig.update_layout(**layout_bar)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Degradation Analysis</div>', unsafe_allow_html=True)

        d_f1  = fed_row["f1"]  - cen_row["f1"]
        d_auc = fed_row["auc"] - cen_row["auc"]

        # tarjetitas resumen para ver cuánto nos ha costado usar federated learning frente a la opción insegura
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1 Variance</div>
            <div class="metric-value" style="color:{'#10B981' if abs(d_f1)<0.03 else '#EF4444'};">
                {d_f1:+.4f}
            </div>
            <div style="font-size:0.8rem;color:#94A3B8;margin-top:4px;">federated vs central</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">AUC Variance</div>
            <div class="metric-value" style="color:{'#10B981' if abs(d_auc)<0.03 else '#EF4444'};">
                {d_auc:+.4f}
            </div>
            <div style="font-size:0.8rem;color:#94A3B8;margin-top:4px;">federated vs central</div>
        </div>
        """, unsafe_allow_html=True)

# pestaña 2: gráficas de convergencia ronda a ronda
with tab2:
    st.markdown('<div class="section-title">Federated Convergence Log</div>', unsafe_allow_html=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("F1 Score Trajectory", "ROC-AUC Trajectory", "Precision Trajectory", "Recall Trajectory"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    #color de los subtítulos de plotly para que no se pierdan en el fondo negro
    for annotation in fig['layout']['annotations']: 
        annotation['font'] = dict(color="#F8FAFC", size=14)

    curve_metrics = [
        ("f1",        1, 1, COLORS["federated"]),
        ("auc",       1, 2, "#8B5CF6"),
        ("precision", 2, 1, "#EC4899"),
        ("recall",    2, 2, "#10B981"),
    ]

    for key, row, col, color in curve_metrics:
        cen_val = cen_row[key]

        # la línea del entrenamiento
        fig.add_trace(go.Scatter(
            x=round_log["round"],
            y=round_log[key],
            mode="lines+markers",
            name=f"FedAvg ({key})",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            showlegend=False,
        ), row=row, col=col)

        # una línea punteada que marca el objetivo (lo que saca el modelo centralizado)
        fig.add_hline(
            y=cen_val,
            line_dash="dash",
            line_color=COLORS["centralized"],
            opacity=0.8,
            row=row, col=col,
            annotation_text=f"Baseline: {cen_val:.3f}",
            annotation_font_size=12,
            annotation_font_color=COLORS["centralized"],
        )

    layout_telemetry = base_layout.copy()
    layout_telemetry.update(height=600, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_layout(**layout_telemetry)
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor="#334155", title_text="Communication Round", title_font=dict(color="#CBD5E1"), tickfont=dict(color="#CBD5E1"), row=i, col=j)
            fig.update_yaxes(gridcolor="#334155", tickfont=dict(color="#CBD5E1"), row=i, col=j)

    st.plotly_chart(fig, use_container_width=True)

# pestaña 3: rendimiento desglosado por cada banco
with tab3:
    st.markdown('<div class="section-title">Node-Level Performance Distribution</div>', unsafe_allow_html=True)

    chosen_metric = st.selectbox(
        "Select optimization metric for breakdown",
        ["f1", "auc", "recall", "precision"],
        format_func=lambda x: x.upper()
    )

    fig = go.Figure()

    for model_name, color in [("Federated", COLORS["federated"]), ("Centralized Baseline", COLORS["centralized"])]:
        match_str = "federated" if model_name == "Federated" else "centralized"
        subset = bank_df[bank_df["model"].str.contains(match_str, case=False, na=False)]
        
        if not subset.empty:
            fig.add_trace(go.Bar(
                name=model_name,
                x=subset["bank"],
                y=subset[chosen_metric],
                marker_color=color,
                text=subset[chosen_metric].map("{:.3f}".format),
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=12)
            ))

    layout_nodes = base_layout.copy()
    layout_nodes.update(
        barmode="group",
        yaxis=dict(range=[0, 1.15], gridcolor="#334155", tickfont=dict(color="#CBD5E1")),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#CBD5E1")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig.update_layout(**layout_nodes)
    st.plotly_chart(fig, use_container_width=True)

# pestaña 4: matrices de confusión para ver falsos positivos/negativos
with tab4:
    st.markdown('<div class="section-title">Global Confusion Matrices</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    def plot_cm(cm_row, title, color):
        tn = cm_row["tn"]; fp = cm_row["fp"]
        fn = cm_row["fn"]; tp = cm_row["tp"]

        z    = [[tn, fp], [fn, tp]]
        text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                [f"FN\n{fn:,}", f"TP\n{tp:,}"]]

        fig = go.Figure(go.Heatmap(
            z=z,
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont={"size": 16, "color": "#FFFFFF"},
            colorscale=[[0, "#1E293B"], [1, color]],
            showscale=False,
            xgap=4,
            ygap=4,
        ))
        
        layout_cm = base_layout.copy()
        layout_cm.update(
            title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
            xaxis=dict(tickvals=[0, 1], ticktext=["Predicted Negative", "Predicted Positive"], side="bottom", tickfont=dict(color="#CBD5E1")),
            yaxis=dict(tickvals=[0, 1], ticktext=["Actual Negative", "Actual Positive"], autorange="reversed", tickfont=dict(color="#CBD5E1")),
            height=360,
            margin=dict(l=80, r=20, t=60, b=60), 
        )
        fig.update_layout(**layout_cm)
        return fig

    for col, (label, color) in zip(
        [col1, col2],
        [("Federated Architecture", COLORS["federated"]), ("Centralized Baseline", COLORS["centralized"])]
    ):
        with col:
            match_str = "federated" if "Federated" in label else "centralized"
            row = cm_df[cm_df["label"].str.contains(match_str, case=False)].iloc[0]
            st.plotly_chart(plot_cm(row, label, color), use_container_width=True)

            # debajo de la matriz ponemos los errores críticos en grande
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
                    <div>
                        <div class="metric-label">False Positives (Type I)</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#EF4444;">{row['fp']:,}</div>
                    </div>
                    <div>
                        <div class="metric-label">False Negatives (Type II)</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#F59E0B;">{row['fn']:,}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# pestaña 5: análisis exploratorio de los datos (eda)
with tab5:
    st.markdown('<div class="section-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    
    col_eda1, col_eda2 = st.columns([1, 1.5])
    
    with col_eda1:
        st.markdown("#### Class Distribution")
        st.markdown("Imbalance problem before local SMOTE applied.")
        
        labels = ['Eligible (Positive)', 'Not Eligible (Negative)']
        values = [12, 88]
        colors = [COLORS['positive'], '#334155']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.5,
            marker=dict(colors=colors, line=dict(color='#0F172A', width=2)),
            textinfo='label+percent',
            textfont=dict(color='#FFFFFF', size=13)
        )])
        
        layout_pie = base_layout.copy()
        layout_pie.update(
            showlegend=False, 
            height=350,
            margin=dict(t=30, b=30, l=10, r=10)
        )
        fig_pie.update_layout(**layout_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_eda2:
        st.markdown("#### Top Features: Mean Differences by Class")
        st.markdown("Most distinguishing variables between eligible and not eligible clients.")
        
        # aquí procesamos el csv de eda de forma dinámica
        # calculamos la diferencia absoluta entre las dos clases
        eda_df['abs_diff'] = abs(eda_df['mean_eligible_1'] - eda_df['mean_not_eligible_0'])
        
        # nos quedamos con las 8 variables con mayor diferencia
        top_eda = eda_df.sort_values('abs_diff', ascending=False).head(8).sort_values('abs_diff', ascending=True)
        
        # pintamos barras agrupadas para comparar la media de los elegibles vs no elegibles
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            y=top_eda['feature'],
            x=top_eda['mean_eligible_1'],
            name='Eligible (1)',
            orientation='h',
            marker_color=COLORS['positive']
        ))
        
        fig_bar.add_trace(go.Bar(
            y=top_eda['feature'],
            x=top_eda['mean_not_eligible_0'],
            name='Not Eligible (0)',
            orientation='h',
            marker_color='#64748B'
        ))
        
        layout_bar_h = base_layout.copy()
        layout_bar_h.update(
            barmode='group',
            xaxis=dict(gridcolor="#334155", title="Average Value", tickfont=dict(color="#CBD5E1")),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#F8FAFC")),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350,
            margin=dict(l=10, r=20, t=30, b=30)
        )
        fig_bar.update_layout(**layout_bar_h)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("""
    <div class="metric-card" style="margin-top: 20px;">
        <div class="metric-label">Dataset Summary</div>
        <p style="color: #CBD5E1; margin-top: 10px; font-size: 0.95rem;">
        The raw dataset contains records distributed across 3 logical nodes. 
        Numerical features were standardized, and categorical variables were one-hot encoded. 
        The significant class imbalance (88% negative / 12% positive) was treated locally at each node using SMOTE 
        to prevent model bias towards the majority class during gradient descent.
        </p>
    </div>
    """, unsafe_allow_html=True)