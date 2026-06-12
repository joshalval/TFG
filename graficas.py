from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from config import DATOS_DIR, get_db_uri


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

SCRIPT_DIR = Path(__file__).resolve().parent
GRAFICAS_DIR = SCRIPT_DIR / "Graficas"
MOSTRAR_GRAFICAS = False
GUARDAR_GRAFICAS = True
FECHA_BASE = pd.Timestamp("2021-01-01")


def preparar_fechas(df):
    df = df.copy()
    df["measured_date"] = pd.to_datetime(df["measured_date"], utc=True, format="mixed").dt.tz_convert("Europe/Madrid")
    fechas_locales = df["measured_date"].dt.tz_localize(None).dt.normalize()
    df["dias"] = (fechas_locales - FECHA_BASE).dt.days
    df["horas"] = df["measured_date"].dt.hour
    return df


def cargar_csv_o_bbdd(nombre_csv, consulta, renombrar=None):
    ruta = DATOS_DIR / nombre_csv
    if ruta.exists():
        df = pd.read_csv(ruta)
    else:
        from sqlalchemy import create_engine

        engine = create_engine(get_db_uri())
        df = pd.read_sql(consulta, engine)

    if renombrar:
        df = df.rename(columns=renombrar)
    return df


def cargar_datos_brutos():
    df = cargar_csv_o_bbdd(
        "ft.csv",
        "SELECT * FROM ft_agricultura;",
        {"temperature": "temperatura", "humidity": "humedad"},
    )
    df = df.dropna(subset=["temperatura", "humedad"])
    df = df[df["temperatura"] <= 38]

    if "id_sensor" not in df.columns:
        df["id_sensor"] = 1

    return preparar_fechas(df).sort_values("measured_date").reset_index(drop=True)


def cargar_agregado_dia():
    df = cargar_csv_o_bbdd(
        "ft_agregada_dias.csv",
        "SELECT * FROM fta_agricultura_1day;",
        {"temperature_mean": "temperatura", "humidity_mean": "humedad"},
    )
    df["measured_date"] = pd.to_datetime(df["measured_date"], utc=True, format="mixed").dt.tz_convert("Europe/Madrid")
    return df


def cargar_agregado_hora():
    df = cargar_csv_o_bbdd(
        "ft_agregada_horas.csv",
        "SELECT * FROM fta_agricultura_1hour;",
        {"temperature_mean": "temperatura", "humidity_mean": "humedad", "aggregation_hour": "horas"},
    )
    return df[df["temperatura"] <= 38]


def nombre_archivo(titulo):
    nombre = titulo.lower()
    for origen, destino in (
        (" ", "_"),
        ("(", ""),
        (")", ""),
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    ):
        nombre = nombre.replace(origen, destino)
    return f"{nombre}.png"


def finalizar_figura(fig, titulo):
    fig.tight_layout()

    if GUARDAR_GRAFICAS:
        GRAFICAS_DIR.mkdir(exist_ok=True)
        fig.savefig(GRAFICAS_DIR / nombre_archivo(titulo), dpi=300, bbox_inches="tight")

    if MOSTRAR_GRAFICAS:
        plt.show()
    else:
        plt.close(fig)


def configurar_eje_x(ax, df, x, xlabel):
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        x_dt = pd.to_datetime(df[x], utc=True, format="mixed").dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
        ax.set_xlim(x_dt.dt.floor("D").min() - pd.Timedelta(days=2), x_dt.dt.floor("D").max() + pd.Timedelta(days=2))
        ax.set_xticks(x_dt.dt.floor("D").unique())
        plt.setp(ax.get_xticklabels(), rotation=90)
    elif x == "dias":
        dias = sorted(pd.to_numeric(df[x].dropna()).astype(int).unique())
        etiquetas = [(FECHA_BASE + pd.Timedelta(days=int(dia))).strftime("%d-%m") for dia in dias]
        ax.set_xticks(dias)
        ax.set_xticklabels(etiquetas, rotation=90)
        ax.set_xlim(min(dias) - 2, max(dias) + 2)
    elif xlabel.lower() == "hora":
        ax.set_xticks(sorted(df[x].dropna().unique()))
    else:
        ax.set_xticks(sorted(df[x].dropna().unique()))
        plt.setp(ax.get_xticklabels(), rotation=90)


def graficar(df, x, y, titulo, xlabel, ylabel, por_sensor=False, tipo="scatter"):
    sensores = sorted(df["id_sensor"].dropna().unique()) if por_sensor and "id_sensor" in df.columns else [None]
    colores = ["blue", "green", "red", "orange", "black"]
    marcadores = ["o", "s", "^", "D", "x"]

    fig, ax = plt.subplots(figsize=(15, 7))

    for i, sensor in enumerate(sensores):
        df_plot = df[df["id_sensor"] == sensor] if sensor is not None else df
        label = f"Sensor {sensor}" if sensor is not None else None
        color = colores[i % len(colores)] if sensor is not None else ("blue" if "temperatura" in y.lower() else "purple")
        marcador = marcadores[i % len(marcadores)]

        if tipo == "plot":
            ax.plot(df_plot[x], df_plot[y], label=label, color=color, marker=marcador)
        else:
            ax.scatter(df_plot[x], df_plot[y], label=label, color=color, marker=marcador)

    if "temperatura" in titulo.lower() and "media" not in titulo.lower():
        ax.set_ylim(14, 40)
    elif "humedad" in titulo.lower() and "media" not in titulo.lower():
        ax.set_ylim(0, 70)

    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    configurar_eje_x(ax, df, x, xlabel)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    finalizar_figura(fig, titulo)


def calcular_agregados_desde_bruto(df):
    return {
        "dia_por_sensor": df.groupby(["id_sensor", "dias"])[["temperatura", "humedad"]].mean().reset_index(),
        "hora_por_sensor": df.groupby(["id_sensor", "horas"])[["temperatura", "humedad"]].mean().reset_index(),
        "dia_global": df.groupby("dias")[["temperatura", "humedad"]].mean().reset_index(),
        "hora_global": df.groupby("horas")[["temperatura", "humedad"]].mean().reset_index(),
    }


def imprimir_correlaciones(df):
    print(f"Correlacion global entre temperatura y humedad: {df['temperatura'].corr(df['humedad']):.3f}")

    tabla = []
    for sensor in sorted(df["id_sensor"].dropna().unique()):
        df_sensor = df[df["id_sensor"] == sensor]
        df_dia = df_sensor.groupby("dias")[["temperatura", "humedad"]].mean()
        df_hora = df_sensor.groupby("horas")[["temperatura", "humedad"]].mean()

        tabla.append({
            "Sensor": sensor,
            "Correlacion diaria": round(df_dia["temperatura"].corr(df_dia["humedad"]), 2),
            "Correlacion horaria": round(df_hora["temperatura"].corr(df_hora["humedad"]), 2),
        })

    print(pd.DataFrame(tabla))


def imprimir_resumen_agregados(df, df_dias, df_horas, agregados):
    grupos_dia = len(agregados["dia_por_sensor"])
    grupos_hora = len(agregados["hora_por_sensor"])
    print(f"Filas en ft_agregada_dias.csv: {len(df_dias)}")
    print(f"Grupos calculados desde bruto por sensor-dia: {grupos_dia}")
    print(f"Filas en ft_agregada_horas.csv: {len(df_horas)}")
    print(f"Grupos calculados desde bruto por sensor-hora: {grupos_hora}")

    if len(df_dias) != grupos_dia:
        print("Aviso: ft_agregada_dias no cubre los mismos grupos que los datos brutos.")
    if len(df_horas) != grupos_hora:
        print("Aviso: ft_agregada_horas no cubre los mismos grupos que los datos brutos.")


def main():
    df = cargar_datos_brutos()
    df_dias = cargar_agregado_dia()
    df_horas = cargar_agregado_hora()
    agregados = calcular_agregados_desde_bruto(df)
    imprimir_resumen_agregados(df, df_dias, df_horas, agregados)

    for var in ("temperatura", "humedad"):
        ylabel = "Temperatura (C)" if var == "temperatura" else "Humedad (%)"

        graficar(df, "measured_date", var, f"{var.capitalize()}", "Fecha", ylabel, por_sensor=True)
        graficar(df, "measured_date", var, f"{var.capitalize()} global", "Fecha", ylabel)
        graficar(df, "dias", var, f"{var.capitalize()} por dias", "Fecha", ylabel)
        graficar(df, "horas", var, f"{var.capitalize()} por horas", "Hora", ylabel)

        graficar(df_dias, "measured_date", var, f"{var.capitalize()} media fta por dias", "Fecha", ylabel, por_sensor=True, tipo="plot")
        graficar(df_horas, "horas", var, f"{var.capitalize()} media fta por horas", "Hora", ylabel, por_sensor=True)

        graficar(agregados["dia_por_sensor"], "dias", var, f"{var.capitalize()} media por dias sensor", "Fecha", ylabel, por_sensor=True, tipo="plot")
        graficar(agregados["hora_por_sensor"], "horas", var, f"{var.capitalize()} media por horas sensor", "Hora", ylabel, por_sensor=True, tipo="plot")
        graficar(agregados["dia_global"], "dias", var, f"{var.capitalize()} media por dias global", "Fecha", ylabel, tipo="plot")
        graficar(agregados["hora_global"], "horas", var, f"{var.capitalize()} media por horas global", "Hora", ylabel, tipo="plot")

    imprimir_correlaciones(df)


if __name__ == "__main__":
    main()
