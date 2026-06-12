import pandas as pd

from config import DATOS_DIR, get_db_uri


def preparar_mediciones(df):
    df = df.rename(columns={"temperature": "temperatura", "humidity": "humedad"}).copy()
    df = df[df["temperatura"] <= 38]
    df["measured_date"] = pd.to_datetime(df["measured_date"], utc=True, format="mixed").dt.tz_convert("Europe/Madrid")
    df["fecha"] = df["measured_date"].dt.date
    df["horas"] = df["measured_date"].dt.hour
    return df


def cargar_mediciones():
    ruta_csv = DATOS_DIR / "ft.csv"
    if ruta_csv.exists():
        print(f"Cargando mediciones desde CSV: {ruta_csv}")
        return preparar_mediciones(pd.read_csv(ruta_csv))

    print("No se ha encontrado Datos/ft.csv. Cargando mediciones desde PostgreSQL.")
    from sqlalchemy import create_engine

    engine = create_engine(get_db_uri())
    return preparar_mediciones(pd.read_sql("SELECT * FROM ft_agricultura;", engine))


def calcular_rangos(df):
    rangos_hora = df.groupby("horas").agg(
        temp_min=("temperatura", "min"),
        temp_max=("temperatura", "max"),
        hum_min=("humedad", "min"),
        hum_max=("humedad", "max"),
    )

    rangos_dia = df.groupby("fecha").agg(
        temp_min=("temperatura", "min"),
        temp_max=("temperatura", "max"),
        hum_min=("humedad", "min"),
        hum_max=("humedad", "max"),
    )

    return rangos_hora, rangos_dia


def etiquetar_punto(temperatura, humedad, rango):
    anomalia_temperatura = int(temperatura < rango["temp_min"] or temperatura > rango["temp_max"])
    anomalia_humedad = int(humedad < rango["hum_min"] or humedad > rango["hum_max"])
    return anomalia_temperatura, anomalia_humedad


def crear_fila(measured_date, temperatura, humedad, rangos_hora, rangos_dia):
    anom_temp_hora, anom_hum_hora = etiquetar_punto(
        temperatura,
        humedad,
        rangos_hora.loc[measured_date.hour],
    )
    anom_temp_dia, anom_hum_dia = etiquetar_punto(
        temperatura,
        humedad,
        rangos_dia.loc[measured_date.date()],
    )

    return {
        "measured_date": measured_date,
        "temperatura": temperatura,
        "humedad": humedad,
        "anomalia_temperatura_horas": anom_temp_hora,
        "anomalia_humedad_horas": anom_hum_hora,
        "anomalia_temperatura_dias": anom_temp_dia,
        "anomalia_humedad_dias": anom_hum_dia,
    }


def crear_puntos_limite_por_grupo(df, columna_grupo, rangos_hora, rangos_dia):
    filas = []
    offsets = [1, 2, -1, -2]

    for _, grupo in df.groupby(columna_grupo):
        fila_max_temp = grupo.loc[grupo["temperatura"].idxmax()]
        fila_min_temp = grupo.loc[grupo["temperatura"].idxmin()]

        for fila_base in (fila_max_temp, fila_min_temp):
            filas.append(
                crear_fila(
                    fila_base["measured_date"],
                    fila_base["temperatura"],
                    fila_base["humedad"],
                    rangos_hora,
                    rangos_dia,
                )
            )

            for offset in offsets:
                filas.append(
                    crear_fila(
                        fila_base["measured_date"],
                        fila_base["temperatura"] + offset,
                        fila_base["humedad"] + offset,
                        rangos_hora,
                        rangos_dia,
                    )
                )

    return pd.DataFrame(filas)


def crear_puntos_limite(df, rangos_hora, rangos_dia):
    puntos_dia = crear_puntos_limite_por_grupo(df, "fecha", rangos_hora, rangos_dia)
    puntos_hora = crear_puntos_limite_por_grupo(df, "horas", rangos_hora, rangos_dia)
    return pd.concat([puntos_dia, puntos_hora], ignore_index=True)


def crear_puntos_manuales(rangos_hora, rangos_dia):
    fechas_locales = [
        "2021-04-29 11:00:00",
        "2021-04-29 02:00:00",
        "2021-04-29 06:00:00",
        "2021-05-03 04:00:00",
        "2021-05-03 00:00:00",
        "2021-05-03 11:00:00",
        "2021-05-03 12:00:00",
        "2021-05-03 12:00:00",
        "2021-05-03 13:00:00",
        "2021-05-03 22:00:00",
        "2021-05-10 11:00:00",
        "2021-05-10 00:00:00",
        "2021-05-10 23:00:00",
        "2021-05-27 00:00:00",
        "2021-05-27 00:00:00",
        "2021-05-27 00:00:00",
        "2021-05-27 00:00:00",
        "2021-05-27 00:00:00",
        "2021-05-27 22:00:00",
        "2021-05-31 00:00:00",
        "2021-05-31 00:00:00",
        "2021-05-31 00:00:00",
        "2021-06-01 00:00:00",
        "2021-06-01 16:00:00",
        "2021-05-02 16:00:00",
        "2021-05-02 16:00:00",
        "2021-05-02 16:00:00",
        "2021-05-02 16:00:00",
        "2021-05-02 16:00:00",
    ]

    puntos = pd.DataFrame({
        "measured_date": pd.Series(pd.to_datetime(fechas_locales)).dt.tz_localize("Europe/Madrid"),
        "temperatura": [
            31, 30, 29, 30, 31, 32, 33, 34, 35, 36,
            32, 33, 34, 30, 31, 32, 33, 34, 35, 33,
            34, 35, 34, 35, 27, 28, 29, 30, 31,
        ],
        "humedad": [
            65.0, 28.0, 60, 50, 40, 34, 35, 29, 33, 55,
            29, 28, 43, 33, 12, 34, 26, 29, 39, 33,
            33, 33, 33, 33, 20, 33, 33, 33, 33,
        ],
    })
    filas = [
        crear_fila(
            fila["measured_date"],
            fila["temperatura"],
            fila["humedad"],
            rangos_hora,
            rangos_dia,
        )
        for _, fila in puntos.iterrows()
    ]
    return pd.DataFrame(filas)


def main():
    df = cargar_mediciones()
    rangos_hora, rangos_dia = calcular_rangos(df)

    puntos_auto = crear_puntos_limite(df, rangos_hora, rangos_dia)
    puntos_manuales = crear_puntos_manuales(rangos_hora, rangos_dia)

    dataset_prueba = pd.concat([puntos_auto, puntos_manuales], ignore_index=True)
    dataset_prueba = dataset_prueba.drop_duplicates(
        subset=[
            "measured_date",
            "temperatura",
            "humedad",
            "anomalia_temperatura_horas",
            "anomalia_humedad_horas",
            "anomalia_temperatura_dias",
            "anomalia_humedad_dias",
        ]
    )
    ruta_salida = DATOS_DIR / "datasetSintetico.csv"
    dataset_prueba.to_csv(ruta_salida, index=False)
    print(f"Dataset sintetico guardado en: {ruta_salida}")


if __name__ == "__main__":
    main()
