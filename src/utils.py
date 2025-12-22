
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import altair as alt


HEADER_ALIASES = {
    "AÑO": ["ANYO", "AÑO", "AÑO ", "ANIO", "ANIO "],
    "MES": ["MES DE ENCARGO", "MES ENCARGO", "MES"],
    "CLIENTE": ["CLIENTE", "CLIENTES"],
    "NOMBRE ENCARGO": ["NOMBRE ENCARGO", "ENCARGO", "TRABAJO", "NOMBRE TRABAJO"],
    "LOCALIDAD": ["LOCALIDAD", "MUNICIPIO", "CIUDAD", "POBLACION", "POBLACIÓN"],
    "TIPO DE CLIENTE": ["TIPO DE CLIENTE", "TIPO DE CLIENTE ", "TIPO DE CLIEN..."],
    "TIPO DE TRABAJO": ["TIPO DE TRABAJO", "TIPO TRABAJO"],
    "CAPTACIÓN CLIENTE": ["CAPTACIÓN CLIENTE", "CAPTACIÓN DE CLIENTE", "CAPTACION CLIENTE", "CAPTACION DE CLIENTE", "CAPTACIN DE CLIENTE", "CAPTACIÓN", "CAPTACION"],
    "MI PRECIO": ["MI PRECIO", "PRECIO", "IMPORTE", "MI PRECIO PRECIO"],
    "ESTADO": ["ESTADO", "PAGADO", "COBRADO", "ESTADO PAGO"],
    "FECHA ENTREGA": ["FECHA ENTREGA", "FECHA ENTREGA ", "FECHA"],
    "HORAS DEDICADAS": ["HORAS DEDICADAS", "HORAS", "H DEDICADAS"],
    "PRECIO/HORA": ["PRECIO/HORA", "PRECIO HORA", "€/H", "EUROS/HORA"],
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().upper()

def find_header_row(df_raw: pd.DataFrame, must_contain: Iterable[str]) -> int:
    """
    Given a raw dataframe (header=None), find the row index that contains
    all the requested tokens (case/space-insensitive).
    """
    tokens = [_norm(t) for t in must_contain]
    for i in range(min(len(df_raw), 100)):  # safety
        row = [_norm(x) for x in df_raw.iloc[i].tolist()]
        joined = " | ".join(row)
        if all(t in joined for t in tokens):
            return i
    raise ValueError(f"Could not find header row containing: {must_contain}")

def _standardize_columns(cols: list[str]) -> list[str]:
    """
    Map messy column names to canonical ones using HEADER_ALIASES.
    Strategy:
      1) Prefer exact alias matches (normalized).
      2) Allow substring matches only for sufficiently-long aliases to avoid collisions
         (e.g. 'CLIENTE' should NOT match 'TIPO DE CLIENTE').
    Unknown columns are kept (sanitized).
    """
    out: list[str] = []
    # Build alias list with lengths (normalized)
    alias_items: list[tuple[str, str]] = []
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            alias_items.append((canon, _norm(a)))
    # Sort so longer aliases match first
    alias_items.sort(key=lambda x: len(x[1]), reverse=True)

    for c in cols:
        cn = _norm(c)
        mapped: Optional[str] = None

        # 1) exact match
        for canon, a in alias_items:
            if cn == a:
                mapped = canon
                break

        # 2) safe substring match (only for long aliases)
        if mapped is None:
            for canon, a in alias_items:
                if len(a) >= 10 and a in cn:
                    mapped = canon
                    break

        if mapped is None:
            mapped = re.sub(r"[^A-Z0-9_/ ]+", "", cn).strip() or "UNKNOWN"
        out.append(mapped)

    # de-duplicate
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for c in out:
        if c not in seen:
            seen[c] = 1
            deduped.append(c)
        else:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
    return deduped

def parse_structured_sheet(
    excel_path: Path,
    sheet_name: str,
    must_contain: Iterable[str] | None = None,
    header_row: int | None = None,
) -> pd.DataFrame:
    """
    Reads an Excel sheet that may contain title rows.
    If header_row is provided, uses it directly as the header.
    Otherwise, tries to find the header row using must_contain.
    """

    # --- CASO 1: header_row explícito  ---
    if header_row is not None:
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=header_row
        )
    else:
        # --- CASO 2: comportamiento actual (detección automática) ---
        if must_contain is None:
            raise ValueError("Either header_row or must_contain must be provided.")

        df_raw = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=None
        )
        header_row = find_header_row(df_raw, must_contain=must_contain)

        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=header_row
        )
    # --- Limpieza común ---
    df.columns = _standardize_columns(list(df.columns))
    
    # Drop fully empty rows
    df = df.dropna(how="all")

    # Drop separator rows (CLIENTE + NOMBRE ENCARGO vacíos)
    if "CLIENTE" in df.columns and "NOMBRE ENCARGO" in df.columns:
        df = df[~(df["CLIENTE"].isna() & df["NOMBRE ENCARGO"].isna())]

    df = df.reset_index(drop=True)
    return df

def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clean_text(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .replace({"nan": np.nan, "NaT": np.nan, "None": np.nan, "": np.nan}))

