from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config import DATOS_DIR, get_db_uri
from graficas import graficar


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

SCRIPT_DIR = Path(__file__).resolve().parent
GRAFICAS_DIR = SCRIPT_DIR / "Graficas_Entropia"
MOSTRAR_GRAFICAS = False
GUARDAR_GRAFICAS = True
TEST_TEMPS = np.arange(1, 51)
EXPERIMENTOS = ("dias", "horas")
METODOS_UMBRAL = ("maximo", "media", "percentil_25")
REPETICIONES_TIEMPO = 5
FECHA_BASE = pd.Timestamp("2021-01-01")


def cargar_entrenamiento_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.rename(columns={"temperature": "temperatura", "humidity": "humedad"})
    return preparar_entrenamiento(df)


def cargar_entrenamiento_bbdd():
    from sqlalchemy import create_engine

    engine = create_engine(get_db_uri())
    consulta = "SELECT * FROM ft_agricultura ORDER BY measured_date;"
    df = pd.read_sql(consulta, engine)
    df = df.rename(columns={"temperature": "temperatura", "humidity": "humedad"})
    return preparar_entrenamiento(df)


def preparar_entrenamiento(df):
    df = df.copy()
    df["measured_date"] = pd.to_datetime(df["measured_date"], utc=True, format="mixed").dt.tz_convert("Europe/Madrid")
    df = df.dropna(subset=["temperatura"])
    df = df[df["temperatura"] <= 38]
    df["fecha"] = df["measured_date"].dt.date
    df["dias"] = (df["measured_date"].dt.tz_localize(None).dt.normalize() - FECHA_BASE).dt.days
    df["horas"] = df["measured_date"].dt.hour
    return df.sort_values("measured_date").reset_index(drop=True)


def cargar_prueba(ruta):
    df = pd.read_csv(ruta)
    df["measured_date"] = pd.to_datetime(df["measured_date"], utc=True, format="mixed").dt.tz_convert("Europe/Madrid")
    df["fecha"] = df["measured_date"].dt.date
    df["dias"] = (df["measured_date"].dt.tz_localize(None).dt.normalize() - FECHA_BASE).dt.days
    df["horas"] = df["measured_date"].dt.hour
    return df.dropna(subset=["temperatura"]).reset_index(drop=True)


def dataset_prueba(df, tiempo):
    etiqueta = etiqueta_real(tiempo)
    conflictos = df.groupby(["temperatura", tiempo])[etiqueta].nunique()
    if (conflictos > 1).any():
        raise ValueError(f"Hay etiquetas contradictorias para puntos temperatura-{tiempo} repetidos.")

    columnas = ["temperatura", tiempo, etiqueta]
    df_limpio = df[columnas].drop_duplicates(["temperatura", tiempo]).copy()
    return df_limpio.sort_values([tiempo, "temperatura"]).reset_index(drop=True)


def cargar_entrenamiento():
    ruta = DATOS_DIR / "ft.csv"
    if ruta.exists():
        print(f"Cargando entrenamiento desde CSV: {ruta}")
        return cargar_entrenamiento_csv(ruta)

    print("Cargando entrenamiento desde PostgreSQL")
    return cargar_entrenamiento_bbdd()


def calcular_entropia(valores):
    frecuencias = pd.Series(valores).value_counts(normalize=True)
    return -np.sum(frecuencias * np.log2(frecuencias))


def calcular_entropia_desde_conteos(conteos):
    total = conteos.sum()
    probabilidades = conteos / total
    return -np.sum(probabilidades * np.log2(probabilidades))


def calcular_entropias(df, tiempo):
    return df.groupby(tiempo)["temperatura"].apply(calcular_entropia).reset_index(name="entropia")


def calcular_variacion_entropia(df, entropias, tiempo):
    entropia_por_tiempo = entropias.set_index(tiempo)["entropia"].to_dict()
    registros = []

    for valor_tiempo, grupo in df.groupby(tiempo):
        conteos_base = grupo["temperatura"].value_counts().to_dict()
        entropia_original = entropia_por_tiempo[valor_tiempo]
        total_nuevo = len(grupo) + 1
        suma_c_log_c = sum(conteo * np.log2(conteo) for conteo in conteos_base.values())

        for temperatura in TEST_TEMPS:
            conteo_actual = conteos_base.get(float(temperatura), 0)
            suma_nueva = suma_c_log_c
            if conteo_actual > 0:
                suma_nueva -= conteo_actual * np.log2(conteo_actual)
            suma_nueva += (conteo_actual + 1) * np.log2(conteo_actual + 1)
            nueva_entropia = np.log2(total_nuevo) - (suma_nueva / total_nuevo)
            registros.append({
                tiempo: valor_tiempo,
                "temperatura": float(temperatura),
                "var_entropia": nueva_entropia - entropia_original,
            })

    return pd.DataFrame(registros)


def calcular_umbral_entropia(var_entropias, tiempo, metodo):
    grupos = var_entropias.groupby(tiempo)["var_entropia"]

    if metodo == "maximo":
        return grupos.max()
    if metodo == "percentil_25":
        return grupos.quantile(0.25)
    if metodo == "media":
        return grupos.mean()

    raise ValueError(f"Metodo de umbral no reconocido: {metodo}")


def predecir_anomalias(df_prueba, var_entropias, tiempo, metodo):
    df_eval = df_prueba.merge(var_entropias, on=[tiempo, "temperatura"], how="left")
    umbral_por_tiempo = calcular_umbral_entropia(var_entropias, tiempo, metodo)
    df_eval["umbral_entropia"] = df_eval[tiempo].map(umbral_por_tiempo)

    if metodo == "maximo":
        prediccion = np.isclose(df_eval["var_entropia"], df_eval["umbral_entropia"])
    else:
        prediccion = df_eval["var_entropia"] >= df_eval["umbral_entropia"]

    df_eval["predicted_label"] = (df_eval["var_entropia"].notna() & prediccion).astype(int)
    return df_eval


def etiqueta_real(tiempo):
    return f"anomalia_temperatura_{tiempo}"


def evaluar(df_eval, tiempo, metodo):
    etiqueta = etiqueta_real(tiempo)
    if etiqueta not in df_eval.columns:
        raise ValueError(f"datasetSintetico.csv debe incluir la columna '{etiqueta}'.")

    y_true = df_eval[etiqueta].astype(int).to_numpy()
    y_pred = df_eval["predicted_label"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Metodo": f"Entropia temperatura por {tiempo} ({metodo})",
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def clasificar_resultado(df_eval, tiempo):
    etiqueta = etiqueta_real(tiempo)
    condiciones = [
        (df_eval[etiqueta] == 1) & (df_eval["predicted_label"] == 1),
        (df_eval[etiqueta] == 0) & (df_eval["predicted_label"] == 0),
        (df_eval[etiqueta] == 0) & (df_eval["predicted_label"] == 1),
        (df_eval[etiqueta] == 1) & (df_eval["predicted_label"] == 0),
    ]
    df_eval["resultado"] = np.select(condiciones, ["TP", "TN", "FP", "FN"], default="Desconocido")
    return df_eval


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


def finalizar_figura(fig, titulo, nombre_archivo_salida=None):
    fig.tight_layout()

    if GUARDAR_GRAFICAS:
        GRAFICAS_DIR.mkdir(exist_ok=True)
        nombre = nombre_archivo_salida or nombre_archivo(titulo)
        fig.savefig(GRAFICAS_DIR / nombre, dpi=300, bbox_inches="tight")

    if MOSTRAR_GRAFICAS:
        plt.show()
    else:
        plt.close(fig)


def formatear_tiempo(valor, tiempo):
    if tiempo == "horas":
        return f"{int(valor):02d}h"

    fecha = FECHA_BASE + pd.Timedelta(days=int(valor))
    return fecha.strftime("%d-%m")


def dibujar_contorno_umbral(ax, var_entropias, tiempo, metodo):
    umbral_por_tiempo = calcular_umbral_entropia(var_entropias, tiempo, metodo)
    datos = var_entropias.copy()
    datos["umbral_entropia"] = datos[tiempo].map(umbral_por_tiempo)
    if metodo == "maximo":
        datos["es_anomalia"] = np.isclose(datos["var_entropia"], datos["umbral_entropia"]).astype(int)
    else:
        datos["es_anomalia"] = (datos["var_entropia"] >= datos["umbral_entropia"]).astype(int)

    valores_tiempo = sorted(datos[tiempo].dropna().unique())
    temperaturas = sorted(datos["temperatura"].dropna().unique())
    if not valores_tiempo or not temperaturas:
        return

    matriz = (
        datos
        .pivot(index="temperatura", columns=tiempo, values="es_anomalia")
        .reindex(index=temperaturas, columns=valores_tiempo)
    )
    xx, yy = np.meshgrid(valores_tiempo, temperaturas)
    ax.contour(xx, yy, matriz.to_numpy(), levels=[0.5], colors="red", linewidths=2)


def nombre_metodo_umbral(metodo):
    nombres = {
        "maximo": "maximo",
        "percentil_25": "percentil 25",
        "media": "media",
    }
    return nombres.get(metodo, metodo)


def nombre_metodo_archivo(metodo):
    return metodo.replace("percentil_25", "percentil25")


def graficar_variacion_entropia(var_entropias, tiempo, valores=None):
    fig, ax = plt.subplots(figsize=(15, 7))
    if valores is None:
        valores = sorted(var_entropias[tiempo].dropna().unique())[:2]

    for valor in valores:
        datos = var_entropias[var_entropias[tiempo] == valor]
        if not datos.empty:
            etiqueta = formatear_tiempo(valor, tiempo)
            ax.plot(datos["temperatura"], datos["var_entropia"], marker="o", label=etiqueta)

    titulo = f"Variacion de entropia por temperatura y {tiempo}"
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel("Temperatura (C)", fontweight="bold")
    ax.set_ylabel("Variacion entropia", fontweight="bold")
    ax.set_xlim(TEST_TEMPS.min() - 1, TEST_TEMPS.max() + 1)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower left")
    finalizar_figura(fig, titulo)


def valores_variacion_representativos(var_entropias, tiempo):
    disponibles = set(var_entropias[tiempo].dropna().unique())
    if tiempo == "horas":
        return [valor for valor in (8, 11, 19) if valor in disponibles]

    fechas = [pd.Timestamp("2021-04-24"), pd.Timestamp("2021-05-19"), pd.Timestamp("2021-06-06")]
    dias = [(fecha - FECHA_BASE).days for fecha in fechas]
    return [dia for dia in dias if dia in disponibles]


def graficar_umbrales_variacion_entropia(var_entropias, tiempo, valores):
    for metodo in METODOS_UMBRAL:
        fig, ax = plt.subplots(figsize=(15, 7))
        umbral_por_tiempo = calcular_umbral_entropia(var_entropias, tiempo, metodo)
        entradas_leyenda = []

        for valor in valores:
            datos = var_entropias[var_entropias[tiempo] == valor]
            if datos.empty:
                continue

            etiqueta = formatear_tiempo(valor, tiempo)
            linea, = ax.plot(datos["temperatura"], datos["var_entropia"], marker="o", label=etiqueta)
            entradas_leyenda.append(linea)
            if valor in umbral_por_tiempo.index:
                linea_umbral = ax.axhline(
                    umbral_por_tiempo.loc[valor],
                    color=linea.get_color(),
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"Umbral {etiqueta}",
                )
                entradas_leyenda.append(linea_umbral)

        titulo = f"Umbral {nombre_metodo_umbral(metodo)} de entropia por temperatura y {tiempo}"
        ax.set_title(titulo, fontsize=14, fontweight="bold")
        ax.set_xlabel("Temperatura (C)", fontweight="bold")
        ax.set_ylabel("Variacion entropia", fontweight="bold")
        ax.set_xlim(TEST_TEMPS.min() - 1, TEST_TEMPS.max() + 1)
        if entradas_leyenda:
            ax.legend(
                entradas_leyenda,
                [h.get_label() for h in entradas_leyenda],
                loc="best",
                ncol=3,
            )

        finalizar_figura(
            fig,
            titulo,
            f"umbral_entropia_temperatura_{tiempo}_{metodo}.png",
        )


def graficar_confusion(df_train_limpio, df_eval, var_entropias, tiempo, metodo, resultados, titulo, nombre_archivo_salida):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.scatter(df_train_limpio[tiempo], df_train_limpio["temperatura"], c="blue", s=50, label="Mediciones correctas")
    dibujar_contorno_umbral(ax, var_entropias, tiempo, metodo)

    estilos = {
        "TN": {"color": "green", "marker": "o", "s": 20, "label": "TN (normales bien detectados)"},
        "FP": {"color": "red", "marker": "x", "s": 80, "label": "FP (normales marcados como anomalos)"},
        "TP": {"color": "red", "marker": "o", "s": 20, "label": "TP (anomalos bien detectados)"},
        "FN": {"color": "green", "marker": "x", "s": 80, "label": "FN (anomalos marcados como normales)"},
    }

    for resultado in resultados:
        datos = df_eval[df_eval["resultado"] == resultado]
        estilo = estilos[resultado]
        ax.scatter(
            datos[tiempo],
            datos["temperatura"],
            c=estilo["color"],
            marker=estilo["marker"],
            s=estilo["s"],
            label=estilo["label"],
        )

    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel("Hora" if tiempo == "horas" else "Dias", fontweight="bold")
    ax.set_ylabel("Temperatura (C)", fontweight="bold")
    if tiempo == "dias":
        dias = sorted(df_train_limpio[tiempo].unique())
        fechas = [FECHA_BASE + pd.Timedelta(days=int(dia)) for dia in dias]
        ax.set_xticks(dias)
        ax.set_xticklabels([fecha.strftime("%d-%m") for fecha in fechas], rotation=90)
        ax.set_xlim(min(dias) - 2, max(dias) + 2)
    else:
        ax.set_xticks(range(24))
        ax.set_xlim(-0.75, 23.75)
    ax.set_ylim(14, 40)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="red", linewidth=2, label="Umbral de entropia"))
    labels.append("Umbral de entropia")
    ax.legend(handles, labels, loc="upper left")
    finalizar_figura(fig, titulo, nombre_archivo_salida)


def generar_graficas_base(df, tiempo, entropias, var_entropias, evaluaciones_por_metodo):
    df_limpio = df[["temperatura", tiempo]].drop_duplicates().sort_values([tiempo, "temperatura"])
    xlabel = "Hora" if tiempo == "horas" else "Dias"

    graficar(df_limpio, tiempo, "temperatura", f"Temperatura por {tiempo}", xlabel, "Temperatura (C)")
    graficar(entropias, tiempo, "entropia", f"Entropia por {tiempo}", xlabel, "Entropia", tipo="plot")
    valores_variacion = valores_variacion_representativos(var_entropias, tiempo)
    graficar_variacion_entropia(var_entropias, tiempo, valores=valores_variacion)
    graficar_umbrales_variacion_entropia(var_entropias, tiempo, valores_variacion)

    for metodo, df_eval in evaluaciones_por_metodo.items():
        nombre_metodo = nombre_metodo_umbral(metodo)
        sufijo_metodo = nombre_metodo_archivo(metodo)
        graficar_confusion(
            df_limpio,
            df_eval,
            var_entropias,
            tiempo,
            metodo,
            ("TN", "FP"),
            f"TN y FP - Temperatura por {tiempo} ({nombre_metodo})",
            f"TN_y_FP_temperatura_por_{tiempo}_{sufijo_metodo}_entropia.png",
        )
        graficar_confusion(
            df_limpio,
            df_eval,
            var_entropias,
            tiempo,
            metodo,
            ("TP", "FN"),
            f"TP y FN - Temperatura por {tiempo} ({nombre_metodo})",
            f"TP_y_FN_temperatura_por_{tiempo}_{sufijo_metodo}_entropia.png",
        )


def medir_tiempo_entropia(df, df_prueba, tiempo, metodo):
    tiempos = []
    for _ in range(REPETICIONES_TIEMPO):
        inicio = perf_counter()
        entropias = calcular_entropias(df, tiempo)
        var_entropias = calcular_variacion_entropia(df, entropias, tiempo)
        predecir_anomalias(df_prueba, var_entropias, tiempo, metodo)
        tiempos.append(perf_counter() - inicio)

    return float(np.median(tiempos))


def ejecutar_experimento(df, df_prueba, tiempo):
    evaluaciones = []
    evaluaciones_por_metodo = {}
    metricas_experimento = []
    df_prueba = dataset_prueba(df_prueba, tiempo)
    entropias = calcular_entropias(df, tiempo)
    var_entropias = calcular_variacion_entropia(df, entropias, tiempo)

    for metodo in METODOS_UMBRAL:
        df_eval = predecir_anomalias(df_prueba, var_entropias, tiempo, metodo)
        tiempo_ejecucion = medir_tiempo_entropia(df, df_prueba, tiempo, metodo)

        df_eval = clasificar_resultado(df_eval, tiempo)
        metricas = evaluar(df_eval, tiempo, metodo)
        metricas["Tiempo_s"] = round(tiempo_ejecucion, 4)
        metricas_experimento.append(metricas)
        evaluaciones.append(df_eval.assign(
            experimento=f"temperatura_{tiempo}",
            metodo_umbral=metodo,
        ))
        evaluaciones_por_metodo[metodo] = df_eval

    generar_graficas_base(df, tiempo, entropias, var_entropias, evaluaciones_por_metodo)
    return metricas_experimento, pd.concat(evaluaciones, ignore_index=True)


def main():
    df = cargar_entrenamiento()
    df_prueba = cargar_prueba(DATOS_DIR / "datasetSintetico.csv")

    resultados = []
    for tiempo in EXPERIMENTOS:
        metricas, _ = ejecutar_experimento(df, df_prueba, tiempo)
        resultados.extend(metricas)

    df_resultados = pd.DataFrame(resultados)
    columnas = ["Metodo", "Tiempo_s", "TP", "FP", "TN", "FN", "Precision", "Recall", "Accuracy", "F1-Score"]
    df_resultados = df_resultados[columnas]
    df_resultados.to_csv(DATOS_DIR / "resultados_entropia.csv", index=False)

    print(df_resultados)
    print(f"Tiempo total entropia: {df_resultados['Tiempo_s'].sum():.4f} s")


if __name__ == "__main__":
    main()
