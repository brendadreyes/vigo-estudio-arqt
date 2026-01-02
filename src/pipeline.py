from __future__ import annotations

from pathlib import Path
import pandas as pd

from .utils import parse_structured_sheet, to_datetime_safe, to_numeric_safe, clean_text

MONTH_MAP = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
    "JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "SETIEMBRE": 9, "OCTUBRE": 10,
    "NOVIEMBRE": 11, "DICIEMBRE": 12
}


def load_trabajos_realizados(excel_path: Path) -> pd.DataFrame:
    df = parse_structured_sheet(
        excel_path=excel_path,
        sheet_name="TRABAJOS REALIZADOS",
        header_row=3  # Fila 4 (0-based index 3)
    )

    # Limpieza de texto
    for col in [
        "CLIENTE", "NOMBRE ENCARGO", "LOCALIDAD", "TIPO DE CLIENTE",
        "TIPO DE TRABAJO", "CAPTACIÓN CLIENTE", "CAPTACIÓN DE CLIENTE", "ESTADO", "MES", "AÑO"
    ]:
        if col in df.columns:
            df[col] = clean_text(df[col])

    # Numéricos
    for col in ["MI PRECIO", "HORAS DEDICADAS", "PRECIO/HORA"]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # (Opcional) si existe FECHA ENTREGA, la parseamos pero NO la usamos para series
    if "FECHA ENTREGA" in df.columns:
        df["FECHA ENTREGA"] = to_datetime_safe(df["FECHA ENTREGA"])

    # Unificar nombre de captación
    if "CAPTACIÓN CLIENTE" not in df.columns and "CAPTACIÓN DE CLIENTE" in df.columns:
        df = df.rename(columns={"CAPTACIÓN DE CLIENTE": "CAPTACIÓN CLIENTE"})

    # --- Normalizar y arrastrar AÑO y MES (celdas combinadas en Excel) ---

    # AÑO
    df["AÑO"] = (
        pd.to_numeric(df["AÑO"], errors="coerce")
        .ffill()
        .astype("Int64")
    )

    # MES (texto)
    df["MES"] = (
        df["MES"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"SETIEMBRE": "SEPTIEMBRE", "NAN": pd.NA, "NONE": pd.NA})
        .ffill()
    )

    # Construir YM_ENCARGO a partir de AÑO + MES
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["MES_NUM"] = df["MES"].map(MONTH_MAP)

    dt = pd.to_datetime(
        dict(year=df["AÑO"], month=df["MES_NUM"], day=1),
        errors="coerce"
    )
    df["YM_ENCARGO"] = dt.dt.to_period("M").astype(str)

    df = df.drop(columns=["MES_NUM"], errors="ignore")

    # Fallback de MI PRECIO si no existe
    if "MI PRECIO" not in df.columns:
        for c in df.columns:
            if "PRECIO" in str(c).upper() and "HORA" not in str(c).upper():
                df["MI PRECIO"] = to_numeric_safe(df[c])
                break

    return df


def load_trabajos_en_curso(excel_path: Path) -> pd.DataFrame:
    df = parse_structured_sheet(
        excel_path=excel_path,
        sheet_name="TRABAJOS EN CURSO",
        header_row=3  # Fila 4 (0-based index 3)
    )

    for col in [
        "AÑO", "MES", "CLIENTE", "NOMBRE ENCARGO", "LOCALIDAD",
        "TIPO DE CLIENTE", "TIPO DE TRABAJO", "CAPTACIÓN CLIENTE", "CAPTACIÓN DE CLIENTE", "ESTADO"
    ]:
        if col in df.columns:
            df[col] = clean_text(df[col])


    if "FECHA ENTREGA" in df.columns:
        df["FECHA ENTREGA"] = to_datetime_safe(df["FECHA ENTREGA"])

    for col in ["MI PRECIO", "HORAS DEDICADAS", "PRECIO/HORA"]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # Unificar captación
    if "CAPTACIÓN CLIENTE" not in df.columns and "CAPTACIÓN DE CLIENTE" in df.columns:
        df = df.rename(columns={"CAPTACIÓN DE CLIENTE": "CAPTACIÓN CLIENTE"})

    # Quitar ENVIADO si existe (da problemas Arrow a veces)
    df = df.drop(columns=["ENVIADO"], errors="ignore")

    # --- Normalizar y arrastrar AÑO y MES (celdas combinadas en Excel) ---

    # AÑO
    df["AÑO"] = (
        pd.to_numeric(df["AÑO"], errors="coerce")
        .ffill()
        .astype("Int64")
    )

    # MES (texto)
    df["MES"] = (
        df["MES"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"SETIEMBRE": "SEPTIEMBRE", "NAN": pd.NA, "NONE": pd.NA})
        .ffill()
    )

    # YM_ENCARGO también para en curso (si lo quieres usar después)
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")
    df["MES_NUM"] = df["MES"].map(MONTH_MAP)
    dt = pd.to_datetime(
        dict(year=df["AÑO"], month=df["MES_NUM"], day=1),
        errors="coerce"
    )
    df["YM_ENCARGO"] = dt.dt.to_period("M").astype(str)
    df = df.drop(columns=["MES_NUM"], errors="ignore")

    return df


def build_metrics(df_realizados: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Return a dictionary of dataframes to feed dashboards:
    - kpis (single row)
    - by_tipo_trabajo
    - by_tipo_cliente
    - by_cliente
    - by_captacion
    - pagos (if ESTADO exists)
    - time_series (monthly by YM_ENCARGO)
    - time_series_entrega (monthly by FECHA ENTREGA)
    """
    out: dict[str, pd.DataFrame] = {}

    # KPIs
    total_trabajos = len(df_realizados)
    total_fact = df_realizados["MI PRECIO"].sum(min_count=1) if "MI PRECIO" in df_realizados.columns else float("nan")
    horas = df_realizados["HORAS DEDICADAS"].sum(min_count=1) if "HORAS DEDICADAS" in df_realizados.columns else float("nan")

    kpis = pd.DataFrame([{
        "trabajos_total": total_trabajos,
        "facturacion_total": total_fact,
        "ingreso_medio_por_trabajo": (total_fact / total_trabajos) if total_trabajos and pd.notna(total_fact) else float("nan"),
        "horas_totales": horas,
        "precio_medio_por_hora": (total_fact / horas) if pd.notna(horas) and horas > 0 else float("nan"),
    }])
    out["kpis"] = kpis

    def _eur_por_hora(x):
        if "HORAS DEDICADAS" not in df_realizados.columns:
            return float("nan")
        h = df_realizados.loc[x.index, "HORAS DEDICADAS"].sum()
        return (x.sum() / h) if h and h > 0 else float("nan")

    # By tipo de trabajo
    if "TIPO DE TRABAJO" in df_realizados.columns:
        g = df_realizados.groupby("TIPO DE TRABAJO", dropna=False).agg(
            trabajos=("NOMBRE ENCARGO", "count"),
            facturacion=("MI PRECIO", "sum"),
            horas=("HORAS DEDICADAS", "sum") if "HORAS DEDICADAS" in df_realizados.columns else ("NOMBRE ENCARGO", "count"),
            ingreso_medio_por_trabajo=("MI PRECIO", "mean"),
            precio_medio_por_hora=("MI PRECIO", _eur_por_hora),
        ).reset_index().sort_values("facturacion", ascending=False)
        out["by_tipo_trabajo"] = g

    # By tipo de cliente
    if "TIPO DE CLIENTE" in df_realizados.columns:
        g = df_realizados.groupby("TIPO DE CLIENTE", dropna=False).agg(
            trabajos=("NOMBRE ENCARGO", "count"),
            facturacion=("MI PRECIO", "sum"),
            horas=("HORAS DEDICADAS", "sum") if "HORAS DEDICADAS" in df_realizados.columns else ("NOMBRE ENCARGO", "count"),
            ingreso_medio_por_trabajo=("MI PRECIO", "mean"),
            precio_medio_por_hora=("MI PRECIO", _eur_por_hora),
        ).reset_index().sort_values("facturacion", ascending=False)
        out["by_tipo_cliente"] = g

    # By cliente
    if "CLIENTE" in df_realizados.columns:
        g = df_realizados.groupby("CLIENTE", dropna=False).agg(
            trabajos=("NOMBRE ENCARGO", "count"),
            facturacion=("MI PRECIO", "sum"),
            horas=("HORAS DEDICADAS", "sum") if "HORAS DEDICADAS" in df_realizados.columns else ("NOMBRE ENCARGO", "count"),
            ingreso_medio_por_trabajo=("MI PRECIO", "mean"),
            precio_medio_por_hora=("MI PRECIO", _eur_por_hora),
        ).reset_index().sort_values("facturacion", ascending=False)
        out["by_cliente"] = g

    # Pagos
    if "ESTADO" in df_realizados.columns and "MI PRECIO" in df_realizados.columns:
        g = df_realizados.groupby("ESTADO", dropna=False).agg(
            trabajos=("NOMBRE ENCARGO", "count"),
            importe=("MI PRECIO", "sum"),
        ).reset_index().sort_values("importe", ascending=False)
        out["pagos"] = g

    # Time series dual: entradas (YM_ENCARGO) vs facturación (YM_ENTREGA)
    if "YM_ENCARGO" in df_realizados.columns:
        entradas = (
            df_realizados.dropna(subset=["YM_ENCARGO"])
            .groupby("YM_ENCARGO", dropna=False)
            .agg(
                encargos_entrados=("NOMBRE ENCARGO", "count"),
                importe_entrado=("MI PRECIO", "sum"),  # 👈 NUEVO: importe total de los encargos que entran ese mes
            )
            .reset_index()
            .rename(columns={"YM_ENCARGO": "YM"})
        )
    else:
        entradas = pd.DataFrame(columns=["YM", "encargos_entrados", "importe_entrado"])
    if "FECHA ENTREGA" in df_realizados.columns and "MI PRECIO" in df_realizados.columns:
        df_realizados['FECHA ENTREGA'] = df_realizados['FECHA ENTREGA'].dt.to_period('M').astype(str)
        fact = (
            df_realizados.dropna(subset=["FECHA ENTREGA"])
            .groupby("FECHA ENTREGA", dropna=False)
            .agg(facturacion_entrega=("MI PRECIO", "sum"))
            .reset_index()
            .rename(columns={"FECHA ENTREGA": "YM"})
        )
    else:
        fact = pd.DataFrame(columns=["YM", "facturacion_entrega"])
    ts_dual = (
    pd.merge(entradas, fact, on="YM", how="outer")
    .fillna({"encargos_entrados": 0, "importe_entrado": 0, "facturacion_entrega": 0})
    )
    ts_dual = ts_dual[ts_dual["YM"].notna() & (ts_dual["YM"].isin(["", "NA", "NaT"]) == False)]

    # Orden cronológico
    if len(ts_dual):
        ts_dual["_YM"] = pd.PeriodIndex(ts_dual["YM"].astype(str), freq="M")
        ts_dual = ts_dual.sort_values("_YM").drop(columns=["_YM"])

    out["time_series_dual"] = ts_dual
    # Captación de cliente
    if "CAPTACIÓN CLIENTE" in df_realizados.columns and "CLIENTE" in df_realizados.columns:
        g = (
            df_realizados.groupby("CAPTACIÓN CLIENTE", dropna=False)
            .agg(
                clientes_unicos=("CLIENTE", "nunique"),   # distintos clientes
                trabajos=("NOMBRE ENCARGO", "count"),     # total trabajos
                facturacion=("MI PRECIO", "sum"),         # opcional
            )
            .reset_index()
            .sort_values("clientes_unicos", ascending=False)
        )
        out["by_captacion"] = g
        print()
    

    return out


def export_artifacts(metrics: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in metrics.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def run_all(excel_path: Path, out_dir: Path) -> None:
    df_realizados = load_trabajos_realizados(excel_path)
    metrics = build_metrics(df_realizados)
    export_artifacts(metrics, out_dir)
