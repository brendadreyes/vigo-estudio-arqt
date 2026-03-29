from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
import numpy as np
from src.pipeline import load_trabajos_realizados, build_metrics
import altair as alt
import plotly.express as px
import tempfile
from pathlib import Path
import uuid
from datetime import datetime

# =========================
# Helpers de formato
# =========================
def money(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):,.0f} €".replace(",", ".")
    except Exception:
        return "—"


def money_2(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):,.2f} €".replace(",", ".")
    except Exception:
        return "—"

def num_1(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):,.1f}".replace(",", ".")
    except Exception:
        return "—"


def safe_str(x) -> str:
    if pd.isna(x):
        return "—"
    return str(x)

def money(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

# =========================
# Cálculos rentabilidad
# =========================
def agg_profitability(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df_in.columns:
        return pd.DataFrame()

    needed = ["MI PRECIO", "HORAS DEDICADAS", "NOMBRE ENCARGO", group_col]
    if any(c not in df_in.columns for c in needed):
        return pd.DataFrame()

    g = (
        df_in.groupby(group_col, dropna=False)
        .agg(
            trabajos=("NOMBRE ENCARGO", "count"),
            facturacion=("MI PRECIO", "sum"),
            horas=("HORAS DEDICADAS", "sum"),
        )
        .reset_index()
    )
    g["eur_h"] = np.where(g["horas"] > 0, g["facturacion"] / g["horas"], np.nan)
    return g


def scatter_fact_vs_eurh(df_agg: pd.DataFrame, label_col: str, title: str):
    if df_agg.empty:
        return None

    return (
        alt.Chart(df_agg)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("eur_h:Q", title="Precio efectivo por hora (€/h)"),
            y=alt.Y("facturacion:Q", title="Facturación total (€)"),
            size=alt.Size("trabajos:Q", title="Nº trabajos"),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="Grupo"),
                alt.Tooltip("trabajos:Q", title="Trabajos"),
                alt.Tooltip("horas:Q", title="Horas", format=",.1f"),
                alt.Tooltip("facturacion:Q", title="Facturación (€)", format=",.0f"),
                alt.Tooltip("eur_h:Q", title="€/h", format=",.2f"),
            ],
        )
        .properties(height=360, title=title)
        .configure_view(stroke=None)
    )


def make_recos(df_agg: pd.DataFrame, label_col: str) -> dict[str, pd.DataFrame]:
    if df_agg.empty:
        return {"priorizar": pd.DataFrame(), "optimizar": pd.DataFrame(), "potenciar": pd.DataFrame()}

    d = df_agg.copy()
    med_fact = d["facturacion"].median(skipna=True) # Mediana de facturación
    med_eurh = d["eur_h"].median(skipna=True) # Mediana de €/h

    priorizar = d[(d["facturacion"] >= med_fact) & (d["eur_h"] >= med_eurh)].sort_values(
        ["facturacion", "eur_h"], ascending=[False, False]
    ) # alta facturación + alto €/h (ambos mayores que la mediana)

    optimizar = d[(d["facturacion"] >= med_fact) & (d["eur_h"] < med_eurh)].sort_values(
        ["facturacion", "eur_h"], ascending=[False, True]
    ) # alta facturación + bajo €/h (mayor honorario que la mediana, pero €/h menor que la mediana)

    potenciar = d[(d["facturacion"] < med_fact) & (d["eur_h"] >= med_eurh)].sort_values(
        ["eur_h", "facturacion"], ascending=[False, False]
    ) # alto €/h pero poco volumen (mayor €/h que la mediana, pero honorario menor que la mediana)

    keep = [label_col, "trabajos", "horas", "facturacion", "eur_h"]
    return {"priorizar": priorizar[keep], "optimizar": optimizar[keep], "potenciar": potenciar[keep]}

def style_fact_eurh(
    styler,
    fact_cmap="Blues",
    eurh_cmap="Greens",
    eurh_reverse=False,
):
    """
    Aplica estilos a Facturación y €/h.
    - eurh_reverse=True → rojo para €/h bajos
    """
    styler = styler.background_gradient(
        subset=["Honorarios"],
        cmap=fact_cmap,
    )

    if eurh_reverse:
        styler = styler.background_gradient(
            subset=["€/h"],
            cmap="Reds_r",  # rojo intenso = bajo €/h
        )
    else:
        styler = styler.background_gradient(
            subset=["€/h"],
            cmap=eurh_cmap,
        )

    return styler
def main():
    st.set_page_config(
    page_title="Dashboard — VIGO Estudio",
    page_icon="assets/logo.jpeg",
    layout="wide"
)
    
    st.markdown(
    """
    <style>
        /* Reducir tamaño del logo del sidebar */
        section[data-testid="stSidebar"] img {
            max-width: 180px;   /* 👈 ajusta aquí */
            height: auto;
            margin: 0 auto 12px auto; /* centrado + espacio abajo */
            display: block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
    st.markdown(
    """
    <style>
        /* Ancho del sidebar */
        section[data-testid="stSidebar"] {
            width: 250px !important;
        }

        /* Ajustar el contenido principal para que no se solape */
        section[data-testid="stSidebar"] + section {
            margin-left: 250px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    with st.sidebar:
        st.image("assets/logo.jpeg", use_container_width=True)

        st.title("📊 Dashboard de análisis")

        st.caption(
        "Análisis integral de la actividad del estudio: "
        "facturación, volumen de trabajos, rentabilidad (€/h), "
        "tipologías de servicios, tipos de cliente, captación y evolución temporal."
    )
        st.divider()

    
        # =========================
        # Carga de datos (persistencia "mejor esfuerzo")
        # =========================

        APP_TMP = Path(tempfile.gettempdir()) / "vigo_estudio_app"
        APP_TMP.mkdir(parents=True, exist_ok=True)

        LAST_EXCEL = APP_TMP / "last_uploaded.xlsx"
        LAST_META = APP_TMP / "last_uploaded_meta.json"


        def _write_last_uploaded(file_bytes: bytes, filename: str) -> None:
            """Sobrescribe el último excel y guarda metadatos."""
            LAST_EXCEL.write_bytes(file_bytes)
            meta = {
                "filename": filename,
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            }
            LAST_META.write_text(pd.Series(meta).to_json(), encoding="utf-8")


        def _read_last_meta() -> dict:
            if LAST_META.exists():
                try:
                    return pd.read_json(LAST_META.read_text(encoding="utf-8"), typ="series").to_dict()
                except Exception:
                    return {}
            return {}


        with st.sidebar:
            st.header("📁 Fuente de datos")
            uploaded = st.file_uploader("Sube el Excel (.xlsx)", type=["xlsx"])

            excel_path: Path | None = None

            if uploaded is not None:
                # Guardamos el último subido (sobrescribe el anterior)
                data = uploaded.getvalue()
                _write_last_uploaded(data, uploaded.name)

                excel_path = LAST_EXCEL
                st.success(f"Excel cargado: {uploaded.name}")

            else:
                # Si no suben nada, intentamos usar el último guardado
                if LAST_EXCEL.exists():
                    meta = _read_last_meta()
                    filename = meta.get("filename", "último_subido.xlsx")
                    st.info(f"Usando último Excel subido: {filename}")
                    excel_path = LAST_EXCEL
                else:
                    st.warning("No hay Excel cargado todavía. Sube un archivo para continuar.")
                    st.stop()

            # Mostrar fecha última carga (preferimos la guardada en meta)
            meta = _read_last_meta()
            if meta.get("uploaded_at"):
                st.caption(f"Última carga: {meta['uploaded_at'].replace('T', ' ')}")
            else:
                # fallback por mtime del archivo
                try:
                    mtime = pd.to_datetime(excel_path.stat().st_mtime, unit="s")
                    st.caption(f"Última carga: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    pass

        
    @st.cache_data(show_spinner=False)
    def _load_df_from_path(path: Path) -> pd.DataFrame:
        return load_trabajos_realizados(path)


    @st.cache_data(show_spinner=False)
    def _load_df_from_bytes(file_bytes: bytes) -> pd.DataFrame:
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / "vigo_uploaded.xlsx"

        tmp_path.write_bytes(file_bytes)

        # Guardamos info en session_state
        st.session_state["_tmp_excel_path"] = str(tmp_path)
        st.session_state["_tmp_excel_uploaded_at"] = datetime.now()

        return load_trabajos_realizados(tmp_path)


    if uploaded is not None:
        df = _load_df_from_bytes(uploaded.getvalue())
    elif excel_path is not None and Path(excel_path).exists():
        df = _load_df_from_path(excel_path)
    else:
        st.stop()
    # -------------------------
    # Filtros (MULTI)
    # -------------------------
    with st.sidebar:

        st.header("🔎 Filtros")
        dff = df.copy()

        anio_sel = []
        if "AÑO" in df.columns:
            opciones_anio = sorted(pd.to_numeric(df["AÑO"], errors="coerce").dropna().astype(int).unique().tolist())
            anio_sel = st.sidebar.multiselect("Año", options=opciones_anio, default=[])

        # --- Filtro Mes (MULTI) ---
        mes_sel = []
        if "MES" in df.columns:
            # Usamos el MES textual tal como viene (ENERO, FEBRERO...)
            opciones_mes = sorted(df["MES"].dropna().astype(str).unique().tolist())
            mes_sel = st.sidebar.multiselect("Mes", options=opciones_mes, default=[])

        tipo_trabajo_sel = []
        if "TIPO DE TRABAJO" in df.columns:
            opciones_tt = sorted(df["TIPO DE TRABAJO"].dropna().astype(str).unique().tolist())
            tipo_trabajo_sel = st.sidebar.multiselect(
                "Tipo de trabajo",
                options=opciones_tt,
                default=[]
            )

        tipo_cliente_sel = []
        if "TIPO DE CLIENTE" in df.columns:
            opciones_tc = sorted(df["TIPO DE CLIENTE"].dropna().astype(str).unique().tolist())
            tipo_cliente_sel = st.sidebar.multiselect(
                "Tipo de cliente",
                options=opciones_tc,
                default=[]
            )

        # Cliente (búsqueda simple)
        cliente_text = st.text_input("Cliente contiene (texto)", value="").strip()

        # Captación / Estado
        capt_sel = []
        if "CAPTACIÓN CLIENTE" in dff.columns:
            opts = sorted(dff["CAPTACIÓN CLIENTE"].dropna().astype(str).unique().tolist())
            capt_sel = st.multiselect("Captación cliente", opts, default=[])

        estado_sel = []
        if "ESTADO" in dff.columns:
            opts = sorted(dff["ESTADO"].dropna().astype(str).unique().tolist())
            estado_sel = st.multiselect("Estado", opts, default=[])
        
        # Aplicar filtros (si no hay selección, no filtra => se queda todo)
        if anio_sel and "AÑO" in dff.columns:
            dff = dff[pd.to_numeric(dff["AÑO"], errors="coerce").isin(anio_sel)]
        if mes_sel and "MES" in dff.columns:
            dff = dff[dff["MES"].astype(str).isin(mes_sel)]
        if tipo_trabajo_sel and "TIPO DE TRABAJO" in dff.columns:
            dff = dff[dff["TIPO DE TRABAJO"].astype(str).isin(tipo_trabajo_sel)]
        if tipo_cliente_sel and "TIPO DE CLIENTE" in dff.columns:
            dff = dff[dff["TIPO DE CLIENTE"].astype(str).isin(tipo_cliente_sel)]
        if cliente_text and "CLIENTE" in dff.columns:
            dff = dff[dff["CLIENTE"].astype(str).str.contains(cliente_text, case=False, na=False)]
        if capt_sel and "CAPTACIÓN CLIENTE" in dff.columns:
            dff = dff[dff["CAPTACIÓN CLIENTE"].astype(str).isin(capt_sel)]
        if estado_sel and "ESTADO" in dff.columns:
            dff = dff[dff["ESTADO"].astype(str).isin(estado_sel)]

            
        st.caption(f"Filas tras filtros: {len(dff):,}".replace(",", "."))



    # ✅ KPIs y métricas SIEMPRE sobre lo filtrado (o todo si no hay filtros)
    metrics = build_metrics(dff)
    total_trab = len(dff)
    total_fact = dff["MI PRECIO"].sum(min_count=1) if "MI PRECIO" in dff.columns else np.nan
    total_h = dff["HORAS DEDICADAS"].sum(min_count=1) if "HORAS DEDICADAS" in dff.columns else np.nan
    eur_h = (total_fact / total_h) if pd.notna(total_fact) and pd.notna(total_h) and total_h > 0 else np.nan
    st.subheader("KPIs")

    kpi_cols = st.columns(5)

    def kpi(title, value):
        st.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:0.9rem; color:#6b7280;">{title}</div>
                <div style="font-size:2.2rem; font-weight:600;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[0]:
        kpi("Clientes", f"{dff['CLIENTE'].nunique():,}".replace(",", "."))

    with kpi_cols[1]:
        kpi("Trabajos", f"{total_trab:,}".replace(",", "."))

    with kpi_cols[2]:
        kpi("Honorarios", money(total_fact))

    with kpi_cols[3]:
        kpi("Horas", f"{int(total_h)}")

    with kpi_cols[4]:
        kpi("Precio efectivo / hora", money_2(eur_h))

    st.caption("Nota: el €/h se calcula como Honorarios / Horas (si horas > 0).")
    st.divider()
    # =========================
    # Pie captación clientes
    # =========================

    st.subheader("Captación de clientes")

    if "by_captacion" in metrics:
        cap = metrics["by_captacion"].copy()

        if cap is None or cap.empty:
            st.info("No hay datos de captación con estos filtros.")
        else:
            # Pie por clientes únicos (si existe) o por trabajos (fallback)
            value_col = "clientes_unicos" if "clientes_unicos" in cap.columns else "trabajos"
            title_val = "Clientes únicos" if value_col == "clientes_unicos" else "Trabajos"

            cap = cap.dropna(subset=["CAPTACIÓN CLIENTE"])
            cap[value_col] = pd.to_numeric(cap[value_col], errors="coerce").fillna(0)
            cap = cap[cap[value_col] > 0].copy()

            if cap.empty:
                st.info("No hay valores > 0 para dibujar el gráfico.")
            else:
                cap = cap.sort_values(value_col, ascending=False)

                # Top N + Otros
                TOP_N = 7
                top = cap.head(TOP_N).copy()
                others = cap.iloc[TOP_N:][value_col].sum()

                if others > 0:
                    top = pd.concat(
                        [top, pd.DataFrame([{"CAPTACIÓN CLIENTE": "Otros", value_col: others}])],
                        ignore_index=True
                    )

                fig = px.pie(
                    top,
                    values=value_col,
                    names="CAPTACIÓN CLIENTE",
                    # title=f"Captación ({title_val})",
                    hole=0.45,  # donut
                )

                # % dentro de cada porción
                fig.update_traces(
                    textposition="inside",
                    textinfo="value+percent",
                    insidetextorientation="radial"
                )

                # Layout limpio
                fig.update_layout(
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend=dict(orientation="v")
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No se encontró la métrica 'by_captacion'.")

    st.divider()

    # =========================
    # Tabs análisis: Tipo trabajo / Tipo cliente / Cliente
    # =========================
    st.header("Análisis")

    tab_tt, tab_tc, tab_cl, tab_temp, tab_strat = st.tabs(["Tipo de trabajo", "Tipo de cliente", "Cliente", "Evolución Temporal", "Acciones (TT x TC)"])
    # -------------------------
    # TAB 1: Tipo de trabajo
    # -------------------------
    with tab_tt:
        st.subheader("Tipo de trabajo: Honorarios vs €/h (tamaño = nº trabajos)")
        by_tt = agg_profitability(dff, "TIPO DE TRABAJO")
        if by_tt.empty:
            st.info("No hay datos suficientes (asegura MI PRECIO y HORAS DEDICADAS, y que exista TIPO DE TRABAJO).")
        else:
            st.altair_chart(scatter_fact_vs_eurh(by_tt, "TIPO DE TRABAJO", "Tipos de trabajo"), use_container_width=True)
            st.caption("Guía rápida: derecha = más €/h · arriba = más honorarios · tamaño = más trabajos")

            with st.expander("Cómo interpretar el gráfico"):
                st.markdown(
                    """
            - **Eje X (€/h):** cuanto más a la **derecha**, más rentable es ese tipo de trabajo (mejor precio por hora).
            - **Eje Y (Honorarios):** cuanto más **arriba**, más dinero total te aporta ese tipo de trabajo.
            - **Tamaño del punto:** cuanto más grande, **más volumen de trabajos** (más encargos de ese tipo).
                    
            **Lectura por zonas (muy útil):**
            - ✅ **Arriba + derecha:** trabajos que aportan mucho y además son rentables → *mantener y potenciar*.
            - 🛠️ **Arriba + izquierda:** aportan dinero, pero con €/h bajo → *revisar precios, tiempos o alcance*.
            - 🚀 **Abajo + derecha:** muy rentables pero con poco volumen → *oportunidad de vender más*.
            - 🧊 **Abajo + izquierda:** poco dinero y poca rentabilidad → *evitar o estandarizar para hacerlos rápido*.
                    """
                )
        

            st.subheader("Rankings")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Top por €/h (más rentable)")
                t = by_tt.sort_values("eur_h", ascending=False).head(15).copy()
                t = t.reset_index(drop=True)
                t = t.rename(columns={"TIPO DE TRABAJO": "Tipo de trabajo", "trabajos": "Nº trabajos",
                                    "horas": "Horas", "facturacion": "Honorarios", "eur_h": "€/h"})
                st.dataframe(
                    t.style
                    .format({
                        "Honorarios": money,
                        "Horas": num_1,
                        "€/h": money_2,
                    })
                    # 👇 SOLO colorea €/h
                    .background_gradient(
                        subset=["€/h"],
                        cmap="Greens"
                    ),
                    width="stretch"
                )
            with c2:
                st.write("Top por Honorarios (más facturación)")
                t = by_tt.sort_values("facturacion", ascending=False).head(15).copy()
                t = t.reset_index(drop=True)
                t = t.rename(columns={"TIPO DE TRABAJO": "Tipo de trabajo", "trabajos": "Nº trabajos",
                                    "horas": "Horas", "facturacion": "Honorarios", "eur_h": "€/h"})
                st.dataframe(
                    t.style
                    .format({
                        "Honorarios": money,
                        "Horas": num_1,
                        "€/h": money_2,
                    })
                    # 👇 SOLO colorea Honorarios
                    .background_gradient(
                        subset=["Honorarios"],
                        cmap="Blues"
                    ),
                    width="stretch"
                )
            
        st.subheader("Acciones sugeridas")
        rec = make_recos(by_tt, "TIPO DE TRABAJO")

        st.write("✅ **Priorizar** (alta facturación + alto €/h)")

        t = rec["priorizar"].rename(columns={
            "TIPO DE TRABAJO": "Tipo de trabajo",
            "trabajos": "Nº trabajos",
            "horas": "Horas",
            "facturacion": "Honorarios",
            "eur_h": "€/h",
        })
        t = t.reset_index(drop=True)
        styler = (
            t.style
            .format({
                "Honorarios": money,
                "Horas": num_1,
                "€/h": money_2,
            })
        )

        styler = style_fact_eurh(
            styler,
            fact_cmap="Blues",
            eurh_cmap="Greens",
            eurh_reverse=False,  # verde = mejor
        )

        st.dataframe(styler, width="stretch")

        st.write("🛠️ **Optimizar / subir precio** (alta facturación + bajo €/h)")

        t = rec["optimizar"].rename(columns={
            "TIPO DE TRABAJO": "Tipo de trabajo",
            "trabajos": "Nº trabajos",
            "horas": "Horas",
            "facturacion": "Honorarios",
            "eur_h": "€/h",
        })
        t = t.reset_index(drop=True)
        styler = (
            t.style
            .format({
                "Honorarios": money,
                "Horas": num_1,
                "€/h": money_2,
            })
        )

        styler = style_fact_eurh(
            styler,
            fact_cmap="Blues",
            eurh_reverse=True,   # 🔥 rojo = €/h bajo
        )

        st.dataframe(styler, width="stretch")


        st.write("🚀 **Potenciar** (alto €/h pero poco volumen)")

        t = rec["potenciar"].rename(columns={
            "TIPO DE TRABAJO": "Tipo de trabajo",
            "trabajos": "Nº trabajos",
            "horas": "Horas",
            "facturacion": "Honorarios",
            "eur_h": "€/h",
        })
        t = t.reset_index(drop=True)
        styler = (
            t.style
            .format({
                "Honorarios": money,
                "Horas": num_1,
                "€/h": money_2,
            })
            .background_gradient(subset=["€/h"], cmap="Greens")
        )

        st.dataframe(styler, width="stretch")
    # -------------------------
    # TAB 2: Tipo de cliente
    # -------------------------

    with tab_tc:
        by_tc = agg_profitability(dff, "TIPO DE CLIENTE")
        if by_tc.empty:
            st.info("No hay datos suficientes (asegura MI PRECIO y HORAS DEDICADAS, y que exista TIPO DE CLIENTE).")
        else:
            # -------------------------
            # Heatmap Demanda trabajos por tipo de cliente
            # -------------------------
            st.subheader("Demanda de trabajos por tipo de cliente")


            cols_needed = {"TIPO DE CLIENTE", "TIPO DE TRABAJO", "NOMBRE ENCARGO"}
            if cols_needed.issubset(dff.columns):

                # Conteo de trabajos
                demand = (
                    dff.groupby(["TIPO DE TRABAJO", "TIPO DE CLIENTE"], dropna=False)
                    .agg(trabajos=("NOMBRE ENCARGO", "count"))
                    .reset_index()
                )

                # Ordenar tipos de trabajo por volumen total (desc)
                order_tt = (
                    demand.groupby("TIPO DE TRABAJO")["trabajos"]
                    .sum()
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )

                # Ordenar tipos de cliente por volumen total (desc)
                order_tc = (
                    demand.groupby("TIPO DE CLIENTE")["trabajos"]
                    .sum()
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )
                demand["rango_trabajos"] = pd.cut(
                demand["trabajos"],
                bins=[0, 1, 3, 6, 10, 100],
                labels=[
                    "1",
                    "2–3",
                    "4–6",
                    "7–10",
                    ">10"
                ]
            )
                
                heatmap = (
                    alt.Chart(demand)
                    .mark_rect(stroke="white", strokeWidth=1)
                    .encode(
                        y=alt.Y(
                            "TIPO DE TRABAJO:N",
                            sort=order_tt,
                            title="Tipo de trabajo",
                            axis=alt.Axis(labelLimit=300)
                        ),
                        x=alt.X(
                            "TIPO DE CLIENTE:N",
                            sort=order_tc,
                            title="Tipo de cliente",
                            axis=alt.Axis(labelAngle= 0)
                        ),
                        color=alt.Color(
                            "rango_trabajos:N",
                            title="Nº de trabajos",
                            scale=alt.Scale(
                                domain=["1", "2–3", "4–6", "7–10", ">10"],
                                range=["#f2f2f2", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
                            ),
                            legend=alt.Legend(orient="right")
                        ),
                        tooltip=[
                            alt.Tooltip("TIPO DE TRABAJO:N", title="Tipo de trabajo"),
                            alt.Tooltip("TIPO DE CLIENTE:N", title="Tipo de cliente"),
                            alt.Tooltip("trabajos:Q", title="Nº trabajos"),
                        ],
                    )
                    .properties(height=max(350, 22 * len(order_tt)))
                    .configure_view(stroke=None)
                )
            
                heatmap = heatmap 
                st.altair_chart(heatmap, use_container_width=True)
                st.caption(
            "Interpretación: filas = tipo de trabajo, columnas = tipo de cliente. "
            "Cuanto más oscuro, más trabajos en esa combinación. Pasa el ratón para ver el número exacto."
        )

            else:
                st.info(
                    "Faltan columnas necesarias: "
                    "'TIPO DE CLIENTE', 'TIPO DE TRABAJO', 'NOMBRE ENCARGO'."
                )
            

            # -------------------------
            # Rankings
            # -------------------------
            st.subheader("Rankings")
            c1, c2 = st.columns(2)

            with c1:
                st.write("Top por €/h (más rentable)")
                t = by_tc.sort_values("eur_h", ascending=False).head(15).copy()
                t = t.reset_index(drop=True)
                t = t.rename(columns={
                    "TIPO DE CLIENTE": "Tipo de cliente",
                    "trabajos": "Nº trabajos",
                    "horas": "Horas",
                    "facturacion": "Honorarios",
                    "eur_h": "€/h",
                })
                st.dataframe(
                    t.style
                    .format({
                        "Honorarios": money,
                        "Horas": num_1,
                        "€/h": money_2,
                    })
                    .background_gradient(subset=["€/h"], cmap="Greens"),
                    width="stretch"
                )

            with c2:
                st.write("Top por Honorarios (más facturación)")
                t = by_tc.sort_values("facturacion", ascending=False).head(15).copy()
                t = t.reset_index(drop=True)
                t = t.rename(columns={
                    "TIPO DE CLIENTE": "Tipo de cliente",
                    "trabajos": "Nº trabajos",
                    "horas": "Horas",
                    "facturacion": "Honorarios",
                    "eur_h": "€/h",
                })
                st.dataframe(
                    t.style
                    .format({
                        "Honorarios": money,
                        "Horas": num_1,
                        "€/h": money_2,
                    })
                    .background_gradient(subset=["Honorarios"], cmap="Blues"),
                    width="stretch"
                )

            # -------------------------
            # Acciones sugeridas
            # -------------------------
            st.subheader("Acciones sugeridas")
            rec = make_recos(by_tc, "TIPO DE CLIENTE")

            # ✅ Priorizar
            st.write("✅ **Priorizar** (alta facturación + alto €/h)")
            t = rec["priorizar"].rename(columns={
                "TIPO DE CLIENTE": "Tipo de cliente",
                "trabajos": "Nº trabajos",
                "horas": "Horas",
                "facturacion": "Honorarios",
                "eur_h": "€/h",
            })
            t = t.reset_index(drop=True)
            styler = (
                t.style
                .format({
                    "Honorarios": money,
                    "Horas": num_1,
                    "€/h": money_2,
                })
            )
            styler = style_fact_eurh(
                styler,
                fact_cmap="Blues",
                eurh_cmap="Greens",
                eurh_reverse=False
            )
            st.dataframe(styler, width="stretch")

            # 🛠️ Optimizar
            st.write("🛠️ **Optimizar / renegociar** (alta facturación + bajo €/h)")
            t = rec["optimizar"].rename(columns={
                "TIPO DE CLIENTE": "Tipo de cliente",
                "trabajos": "Nº trabajos",
                "horas": "Horas",
                "facturacion": "Honorarios",
                "eur_h": "€/h",
            })
            t = t.reset_index(drop=True)
            styler = (
                t.style
                .format({
                    "Honorarios": money,
                    "Horas": num_1,
                    "€/h": money_2,
                })
            )
            styler = style_fact_eurh(
                styler,
                fact_cmap="Blues",
                eurh_reverse=True  # rojo = €/h bajo
            )
            st.dataframe(styler, width="stretch")

            # 🚀 Potenciar
            st.write("🚀 **Potenciar** (alto €/h pero poco volumen)")
            t = rec["potenciar"].rename(columns={
                "TIPO DE CLIENTE": "Tipo de cliente",
                "trabajos": "Nº trabajos",
                "horas": "Horas",
                "facturacion": "Honorarios",
                "eur_h": "€/h",
            })
            t = t.reset_index(drop=True)
            styler = (
                t.style
                .format({
                    "Honorarios": money,
                    "Horas": num_1,
                    "€/h": money_2,
                })
                .background_gradient(subset=["€/h"], cmap="Greens")
            )
            st.dataframe(styler, width="stretch")

    
    # -------------------------
    # TAB 3: Cliente
    # -------------------------

    with tab_cl:
        
        by_cl = agg_profitability(dff, "CLIENTE")
        if by_cl.empty:
            st.info("No hay datos suficientes (asegura MI PRECIO y HORAS DEDICADAS, y que exista CLIENTE).")
        else:
        
            # -------------------------
            # Rentabilidad y volumen por cliente
            # -------------------------
            by_cl = agg_profitability(dff, "CLIENTE")
            if by_cl.empty:
                st.info("No hay datos suficientes (asegura MI PRECIO y HORAS DEDICADAS, y que exista CLIENTE).")
            else:
                # Renombres comunes
                def prep(df: pd.DataFrame) -> pd.DataFrame:
                    t = df.copy().reset_index(drop=True)
                    return t.rename(columns={
                        "CLIENTE": "Cliente",
                        "trabajos": "Nº trabajos",
                        "horas": "Horas",
                        "facturacion": "Honorarios",
                        "eur_h": "€/h",
                    })

                # =========================
                # 1) Rentabilidad (€/h)
                # =========================
                st.subheader("Rentabilidad por cliente (€/h)")

                c1, c2 = st.columns(2)

                with c1:
                    st.write("✅ Clientes más rentables (€/h alto)")
                    t = by_cl.sort_values("eur_h", ascending=False).head(15).copy()
                    t = prep(t)

                    st.dataframe(
                        t.style
                        .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                        .background_gradient(subset=["€/h"], cmap="Greens")
                        .background_gradient(subset=["Honorarios"], cmap="Blues"),
                        width="stretch"
                    )

                with c2:
                    st.write("⚠️ Clientes menos rentables (€/h bajo)")
                    t = by_cl.sort_values("eur_h", ascending=True).head(15).copy()
                    t = prep(t)

                    st.dataframe(
                        t.style
                        .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                        # rojo fuerte para €/h bajo
                        .background_gradient(subset=["€/h"], cmap="Reds_r")
                        .background_gradient(subset=["Honorarios"], cmap="Blues"),
                        width="stretch"
                    )

                # =========================
                # 2) Volumen (Honorarios / Trabajos)
                # =========================
                st.subheader("Volumen por cliente")

                c3, c4 = st.columns(2)

                with c3:
                    st.write("💰 Clientes que más facturan (Honorarios)")
                    t = by_cl.sort_values("facturacion", ascending=False).head(15).copy()
                    t = prep(t)

                    st.dataframe(
                        t.style
                        .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                        .background_gradient(subset=["Honorarios"], cmap="Blues"),
                        width="stretch"
                    )

                with c4:
                    st.write("🧱 Clientes con más carga (Nº trabajos)")
                    t = by_cl.sort_values("trabajos", ascending=False).head(15).copy()
                    t = prep(t)

                    st.dataframe(
                        t.style
                        .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                        .background_gradient(subset=["Nº trabajos"], cmap="Purples"),
                        width="stretch"
                    )

                st.caption("Nota: el €/h se calcula como Honorarios / Horas (si horas > 0).")



    # -------------------------
    # TAB 4: Temporal
    # -------------------------
    with tab_temp:
        st.subheader("Entradas vs Facturación (barras: encargos entrantes · línea: facturación por entrega)")
        if "time_series_dual" in metrics:
            ts = metrics["time_series_dual"].copy()

            if ts.empty:
                st.info("No hay datos suficientes para construir la serie temporal.")
            else:
                # Orden cronológico correcto
                ts["_YM"] = pd.PeriodIndex(ts["YM"].astype(str), freq="M")
                ts = ts.sort_values("_YM")

                base = alt.Chart(ts).encode(
                    x=alt.X(
                        "YM:N",
                        sort=ts["YM"].tolist(),
                        title="Año-Mes",
                        axis=alt.Axis(labelAngle= -45)
                    )
                )

                bars = base.mark_bar(opacity=0.35).encode(
                    y=alt.Y("encargos_entrados:Q", title="Encargos entrados"),
                    tooltip=[
                        alt.Tooltip("YM:N", title="Año-Mes"),
                        alt.Tooltip("encargos_entrados:Q", title="Encargos entrados"),
                        alt.Tooltip("importe_entrado:Q", title="Importe entrado (€)", format=",.0f"),
                        alt.Tooltip("facturacion_entrega:Q", title="Facturación (€)", format=",.0f"),
                    ],
                )

                line = base.mark_line(point=True, strokeWidth=3).encode(
                    y=alt.Y(
                        "facturacion_entrega:Q",
                        title="Facturación (€)",
                        axis=alt.Axis(orient="right"),
                    ),
                    tooltip=[
                        alt.Tooltip("YM:N", title="Año-Mes"),
                        alt.Tooltip("encargos_entrados:Q", title="Encargos entrados"),
                        alt.Tooltip("importe_entrado:Q", title="Importe entrado (€)", format=",.0f"),
                        alt.Tooltip("facturacion_entrega:Q", title="Facturación (€)", format=",.0f"),
                    ],
                )

                chart = (
                    alt.layer(bars, line)
                    .resolve_scale(y="independent")
                    .properties(height=340)
                    .configure_view(stroke=None)
                )

                st.altair_chart(chart, use_container_width=True)

        else:
            st.info("No se pudo construir la serie temporal dual (time_series_dual).")

        st.subheader("Facturación por año y tipo de cliente")

        # 1) Agregación
        df_year_tt = (
            dff.dropna(subset=["AÑO", "TIPO DE CLIENTE", "MI PRECIO"])
            .groupby(["AÑO", "TIPO DE CLIENTE"], dropna=False)
            .agg(facturacion=("MI PRECIO", "sum"))
            .reset_index()
        )

        if df_year_tt.empty:
            st.info("No hay datos suficientes (AÑO, TIPO DE TRABAJO y MI PRECIO).")
        else:
            # Asegura tipos y orden de años
            df_year_tt["AÑO"] = df_year_tt["AÑO"].astype(int).astype(str)

            years = sorted(df_year_tt["AÑO"].unique().tolist())

            # 2) Gráfico barras agrupadas (dodge)
            chart = (
                alt.Chart(df_year_tt)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "AÑO:N",
                        sort=years,
                        title="Año",
                        axis=alt.Axis(labelAngle=0)
                    ),
                    xOffset=alt.XOffset("TIPO DE CLIENTE:N"),   # 👈 barras lado a lado
                    y=alt.Y("facturacion:Q", title="Facturación (€)"),
                    color=alt.Color("TIPO DE CLIENTE:N", title="Tipo de cliente"),
                    tooltip=[
                        alt.Tooltip("AÑO:N", title="Año"),
                        alt.Tooltip("TIPO DE CLIENTE:N", title="Tipo de cliente"),
                        alt.Tooltip("facturacion:Q", title="Facturación (€)", format=",.0f"),
                    ],
                )
                .properties(height=420)
                .configure_view(stroke=None)
            )

            st.altair_chart(chart, use_container_width=True)

    # -------------------------
    # Acciones estratégicas (Tipo trabajo x Tipo cliente)
    # -------------------------
    with tab_strat:
        st.subheader("🧠 Acciones estratégicas: Tipo de trabajo × Tipo de cliente")
        st.caption(
            "Agrupamos cada combinación (tipo de trabajo + tipo de cliente) como una unidad de negocio. "
            "Clasificamos según volumen (Honorarios) y rentabilidad (€/h) usando la mediana como umbral."
        )
        cols_needed = {"TIPO DE TRABAJO", "TIPO DE CLIENTE", "NOMBRE ENCARGO", "HORAS DEDICADAS", "MI PRECIO"}
        if not cols_needed.issubset(dff.columns):
            st.info("Faltan columnas necesarias para este análisis (TT, TC, MI PRECIO, HORAS DEDICADAS, NOMBRE ENCARGO).")
        else:
            tmp = dff.copy()

            # Asegurar numéricos
            tmp["MI PRECIO"] = pd.to_numeric(tmp["MI PRECIO"], errors="coerce")
            tmp["HORAS DEDICADAS"] = pd.to_numeric(tmp["HORAS DEDICADAS"], errors="coerce")

            by_tt_tc = (
                tmp.dropna(subset=["TIPO DE TRABAJO", "TIPO DE CLIENTE"])
                .groupby(["TIPO DE TRABAJO", "TIPO DE CLIENTE"], dropna=False)
                .agg(
                    trabajos=("NOMBRE ENCARGO", "count"),
                    horas=("HORAS DEDICADAS", "sum"),
                    facturacion=("MI PRECIO", "sum"),
                )
                .reset_index()
            )

            if by_tt_tc.empty:
                st.info("No hay datos suficientes con la selección actual.")
            else:
                # €/h robusto
                by_tt_tc["eur_h"] = np.where(by_tt_tc["horas"] > 0, by_tt_tc["facturacion"] / by_tt_tc["horas"], np.nan)

                # Quitamos combinaciones sin horas o sin facturación (opcional)
                by_tt_tc = by_tt_tc.replace([np.inf, -np.inf], np.nan)

                fact_med = by_tt_tc["facturacion"].median(skipna=True)
                eurh_med = by_tt_tc["eur_h"].median(skipna=True)

                def classify(row):
                    if pd.isna(row["facturacion"]) or pd.isna(row["eur_h"]):
                        return "Sin datos"
                    if row["facturacion"] >= fact_med and row["eur_h"] >= eurh_med:
                        return "Escalar"
                    if row["facturacion"] >= fact_med and row["eur_h"] < eurh_med:
                        return "Revisar"
                    if row["facturacion"] < fact_med and row["eur_h"] >= eurh_med:
                        return "Oportunidad"
                    return "Evitar"

                by_tt_tc["accion"] = by_tt_tc.apply(classify, axis=1)

                # Helper formato tabla
                def prep_table(df: pd.DataFrame) -> pd.DataFrame:
                    t = df.copy().reset_index(drop=True)
                    return t.rename(columns={
                        "TIPO DE TRABAJO": "Tipo de trabajo",
                        "TIPO DE CLIENTE": "Tipo de cliente",
                        "trabajos": "Nº trabajos",
                        "horas": "Horas",
                        "facturacion": "Honorarios",
                        "eur_h": "€/h",
                    })

                # Contadores resumen
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Escalar", int((by_tt_tc["accion"] == "Escalar").sum()))
                c2.metric("Revisar", int((by_tt_tc["accion"] == "Revisar").sum()))
                c3.metric("Oportunidad", int((by_tt_tc["accion"] == "Oportunidad").sum()))
                c4.metric("Evitar", int((by_tt_tc["accion"] == "Evitar").sum()))

                st.divider()

                # 1) Escalar
                st.write("✅ **Escalar** — alto volumen y alta rentabilidad")
                t = prep_table(by_tt_tc[by_tt_tc["accion"] == "Escalar"].sort_values("facturacion", ascending=False))
                t.drop(columns=["accion"], inplace=True)
                st.dataframe(
                    t.style
                    .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                    .background_gradient(subset=["Honorarios"], cmap="Blues")
                    .background_gradient(subset=["€/h"], cmap="Greens"),
                    width="stretch"
                )

                # 2) Revisar
                st.write("🛠️ **Revisar precio/tiempos** — alto volumen, €/h bajo")
                t = prep_table(by_tt_tc[by_tt_tc["accion"] == "Revisar"].sort_values("facturacion", ascending=False))
                t.drop(columns=["accion"], inplace=True)
                st.dataframe(
                    t.style
                    .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                    .background_gradient(subset=["Honorarios"], cmap="Blues")
                    .background_gradient(subset=["€/h"], cmap="Reds_r"),
                    width="stretch"
                )

                # 3) Oportunidad
                st.write("🎯 **Oportunidad** — €/h alto pero poco volumen")
                t = prep_table(by_tt_tc[by_tt_tc["accion"] == "Oportunidad"].sort_values("eur_h", ascending=False))
                t.drop(columns=["accion"], inplace=True)
                st.dataframe(
                    t.style
                    .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                    .background_gradient(subset=["€/h"], cmap="Greens"),
                    width="stretch"
                )

                # 4) Evitar
                st.write("❌ **Evitar / estandarizar** — bajo impacto y €/h bajo")
                t = prep_table(by_tt_tc[by_tt_tc["accion"] == "Evitar"].sort_values(["facturacion", "eur_h"], ascending=[False, True]))
                t.drop(columns=["accion"], inplace=True)
                st.dataframe(
                    t.style
                    .format({"Honorarios": money, "Horas": num_1, "€/h": money_2})
                    .background_gradient(subset=["€/h"], cmap="Reds_r"),
                    width="stretch"
                )

                # Opcional: mostrar también "Sin datos"
                if (by_tt_tc["accion"] == "Sin datos").any():
                    st.write("ℹ️ **Sin datos** — falta horas o facturación")
                    t = prep_table(by_tt_tc[by_tt_tc["accion"] == "Sin datos"])
                    st.dataframe(t, width="stretch")

    # =========================
    # Detalle (opcional)
    # =========================
    with st.expander("🔍 Ver detalle de trabajos (según selección)", expanded=False):
        dview = dff.copy()
        for col in ["YM_ENCARGO","YM_ENTREGA"]: 
            if col in dview.columns:
                dview["_YM"] = pd.PeriodIndex(dview[col].astype(str), freq="M")
                dview = dview.sort_values("_YM", ascending=False).drop(columns=["_YM"])
        dview = dview.rename(columns={"YM_ENCARGO": "FECHA DE ENCARGO", "YM_ENTREGA": "FECHA DE ENTREGA"})
        
        st.dataframe(dview.drop(columns=["UNNAMED 0","UNNAMED 11"], errors="ignore"), width="stretch")
    st.caption("Nota: Las métricas se recalculan automáticamente al cargar/actualizar el Excel. No requiere ejecutar código.")






if __name__ == "__main__":
    main()