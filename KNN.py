from pathlib import Path
import sys
from time import perf_counter

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors

from config import DATOS_DIR, get_db_uri


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

SCRIPT_DIR = Path(__file__).resolve().parent
FECHA_BASE = pd.Timestamp("2021-01-01")

METODOS = ("percentil_95", "media_varianza", "maximo_promedio", "temporalidad_distinta")
N_VECINOS = (3, 5, 7)
EXPERIMENTOS = (("temperatura", "dias"), ("temperatura", "horas"))
REPETICIONES_TIEMPO = 5
MOSTRAR_GRAFICAS = False
GUARDAR_GRAFICAS = True
MOSTRAR_MUESTRA = False
GRAFICAS_DIR = SCRIPT_DIR / "Graficas_KNN"


def crear_engine():
    from sqlalchemy import create_engine

    return create_engine(get_db_uri())


def preparar_fechas(df):
    df = df.copy()
    df["measured_date"] = pd.to_datetime(df["measured_date"], utc=True, format="mixed").dt.tz_convert("Europe/Madrid")
    fechas_locales = df["measured_date"].dt.tz_localize(None).dt.normalize()
    df["dias"] = (fechas_locales - FECHA_BASE).dt.days
    df["horas"] = df["measured_date"].dt.hour
    return df.dropna(subset=["temperatura", "humedad"])


def cargar_entrenamiento(engine):
    consulta = "SELECT * FROM ft_agricultura;"
    df = pd.read_sql(consulta, engine)
    df = df.rename(columns={"temperature": "temperatura", "humidity": "humedad"})
    df = df[df["temperatura"] <= 38]
    return preparar_fechas(df).sort_values("measured_date").reset_index(drop=True)


def cargar_entrenamiento_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.rename(columns={"temperature": "temperatura", "humidity": "humedad"})
    df = df[df["temperatura"] <= 38]
    return preparar_fechas(df).sort_values("measured_date").reset_index(drop=True)


def cargar_prueba(ruta):
    df = pd.read_csv(ruta)
    return preparar_fechas(df).reset_index(drop=True)


def dataset_prueba(df, var, tiempo):
    columna_etiqueta = f"anomalia_{var}_{tiempo}"
    conflictos = df.groupby([var, tiempo])[columna_etiqueta].nunique()
    if (conflictos > 1).any():
        raise ValueError(f"Hay etiquetas contradictorias para puntos {var}-{tiempo} repetidos.")

    columnas = [var, tiempo, columna_etiqueta]
    df_limpio = df[columnas].drop_duplicates([var, tiempo]).copy()
    return df_limpio.sort_values([tiempo, var]).reset_index(drop=True)


def extender_horas_ciclicas(df):
    partes = [df]
    partes.extend(df[df["horas"] == h].assign(horas=h + 24) for h in range(0, 6))
    partes.extend(df[df["horas"] == h].assign(horas=h - 24) for h in range(23, 18, -1))
    return pd.concat(partes, ignore_index=True)


def dataset_entrenamiento(df, var, tiempo):
    columnas = [tiempo, var]
    df_limpio = df.drop_duplicates(columnas).copy()
    if tiempo == "horas":
        df_limpio = extender_horas_ciclicas(df_limpio)
    return df_limpio.reset_index(drop=True)


def matriz_puntos(df, var, tiempo):
    return df[[var, tiempo]].to_numpy(dtype=float)


def vecinos_temporalidad_distinta(x_ref, punto, k, idx_ignorar=None):
    valor_actual = punto[0]
    tiempo_actual = int(punto[1])
    distancias = np.linalg.norm(x_ref - punto, axis=1)
    indices_ordenados = np.argsort(distancias)

    tiempos_usados = set()
    vecinos = []
    mismo_tiempo_usado = False

    for idx in indices_ordenados:
        if idx_ignorar is not None and idx == idx_ignorar:
            continue

        valor_vecino = x_ref[idx][0]
        tiempo_vecino = int(x_ref[idx][1])

        if tiempo_vecino == tiempo_actual and valor_vecino != valor_actual and not mismo_tiempo_usado:
            vecinos.append((distancias[idx], idx))
            mismo_tiempo_usado = True
        elif tiempo_vecino != tiempo_actual and tiempo_vecino not in tiempos_usados:
            vecinos.append((distancias[idx], idx))
            tiempos_usados.add(tiempo_vecino)

        if len(vecinos) == k:
            break

    return vecinos


def filtrar_vecinos_knn(distancias, indices):
    pares = list(zip(distancias, indices))
    filtrados = [(dist, idx) for dist, idx in pares if dist > 0]
    return filtrados if len(filtrados) < len(pares) else filtrados[:-1]


def distancias_medias_knn(knn, puntos):
    distancias, indices = knn.kneighbors(puntos)
    medias = [
        np.mean([dist for dist, _ in filtrar_vecinos_knn(dist, idx)])
        for dist, idx in zip(distancias, indices)
    ]
    vecinos = [
        filtrar_vecinos_knn(dist, idx)
        for dist, idx in zip(distancias, indices)
    ]
    return np.array(medias), vecinos


def calcular_umbral(metodo, distancias_train):
    if metodo == "percentil_95":
        return np.percentile(distancias_train, 95)
    if metodo == "media_varianza":
        return distancias_train.mean() + 3 * distancias_train.std()
    if metodo == "maximo_promedio":
        return distancias_train.max()
    if metodo == "temporalidad_distinta":
        return distancias_train.mean()
    raise ValueError(f"Metodo no reconocido: {metodo}")


def media_distancias(vecinos):
    return np.mean([dist for dist, _ in vecinos]) if vecinos else np.inf


def entrenar_y_predecir(x_train, x_prueba, metodo, k):
    if metodo == "temporalidad_distinta":
        dist_train = np.array([
            media_distancias(vecinos_temporalidad_distinta(x_train, punto, k, i))
            for i, punto in enumerate(x_train)
        ])
        vecinos_prueba = [vecinos_temporalidad_distinta(x_train, punto, k) for punto in x_prueba]
        dist_prueba = np.array([media_distancias(vecinos) for vecinos in vecinos_prueba])
    else:
        knn = NearestNeighbors(n_neighbors=k + 1).fit(x_train)
        dist_train, _ = distancias_medias_knn(knn, x_train)
        dist_prueba, vecinos_prueba = distancias_medias_knn(knn, x_prueba)

    umbral = calcular_umbral(metodo, dist_train)
    return umbral, dist_prueba, vecinos_prueba, dist_prueba > umbral


def etiqueta_tiempo(valor, tiempo):
    if tiempo == "horas":
        return f"{int(round(valor)) % 24:02d}h"
    return (FECHA_BASE + pd.Timedelta(days=valor)).strftime("%d-%m")


def imprimir_muestra(var, tiempo, x_train, x_prueba, vecinos, distancias, umbral, anomalias, max_puntos=10):
    for i, punto in enumerate(x_prueba[:max_puntos]):
        print(f"Punto {i + 1} ({punto[0]:.1f}, {etiqueta_tiempo(punto[1], tiempo)})")

        for j, (dist, idx) in enumerate(vecinos[i]):
            vecino = x_train[idx]
            print(
                f"  Vecino {j + 1}: ({vecino[0]:.1f}, {etiqueta_tiempo(vecino[1], tiempo)}), "
                f"Distancia: {dist:.4f}"
            )

        print(f"Distancia media: {distancias[i]:.4f}, Umbral: {umbral:.4f}")
        print("Anomalo?", "Si" if anomalias[i] else "No")
        print("-" * 50)


def imprimir_ejemplo_distribucion(df_entrenamiento, var, tiempo, punto, descripcion, k):
    df_train = dataset_entrenamiento(df_entrenamiento, var, tiempo)
    x_train = matriz_puntos(df_train, var, tiempo)
    etiqueta_temporal = "hora" if tiempo == "horas" else "dia"

    print("\n" + "=" * 72)
    print(f"Ejemplo KNN - {var.capitalize()} por {tiempo}")
    print("=" * 72)
    print(f"Punto evaluado: {descripcion}")
    print(f"Numero de vecinos: {k}")

    for metodo in METODOS:
        umbral, distancias, vecinos, anomalias = entrenar_y_predecir(x_train, punto, metodo, k)
        print("\n" + "-" * 72)
        print(f"Metodo: {metodo}")
        print("Vecinos detectados:")

        for i, (distancia, indice) in enumerate(vecinos[0], start=1):
            vecino = x_train[indice]
            fecha_registro = df_train.iloc[indice]["measured_date"].strftime("%d-%m %Hh")
            print(
                f"  Vecino {i}: {vecino[0]:.1f} C, "
                f"{etiqueta_temporal} {etiqueta_tiempo(vecino[1], tiempo)}, "
                f"registro {fecha_registro}, "
                f"distancia {distancia:.4f}"
            )

        print(f"Distancia media: {distancias[0]:.4f}")
        print(f"Umbral: {umbral:.4f}")
        print("Decision:", "ANOMALA" if anomalias[0] else "NORMAL")


def imprimir_ejemplo_consola(df_entrenamiento):
    var = "temperatura"
    k = 5
    fecha = pd.Timestamp("2021-05-10")
    dia = (fecha.normalize() - FECHA_BASE).days

    print("=" * 72)
    print("Ejemplo KNN para explicar los cuatro metodos de umbral")
    print("=" * 72)
    print("Nota: el punto se crea manualmente para explicar el funcionamiento.")

    imprimir_ejemplo_distribucion(
        df_entrenamiento,
        var,
        "dias",
        np.array([[35.0, dia]], dtype=float),
        "35.0 C el dia 10-05",
        k,
    )
    imprimir_ejemplo_distribucion(
        df_entrenamiento,
        var,
        "horas",
        np.array([[35.0, 23.0]], dtype=float),
        "35.0 C a las 23h (la distribucion horaria no utiliza el dia)",
        k,
    )
    print("=" * 72)


def crear_malla(df_train, var, tiempo):
    x_min, x_max = df_train[tiempo].min(), df_train[tiempo].max()
    y_min, y_max = df_train[var].min(), df_train[var].max()
    margen_x = 3 if tiempo == "horas" else 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min - margen_x, x_max + margen_x, 200),
        np.linspace(y_min - 3, y_max + 3, 100),
    )
    return xx, yy, np.c_[yy.ravel(), xx.ravel()]


def calcular_distancia_malla(x_train, grid_points, metodo, k):
    if metodo == "temporalidad_distinta":
        distancias = [
            media_distancias(vecinos_temporalidad_distinta(x_train, np.round(punto).astype(int), k))
            for punto in grid_points
        ]
        return np.array(distancias)

    knn = NearestNeighbors(n_neighbors=k + 1).fit(x_train)
    distancias, indices = knn.kneighbors(np.round(grid_points).astype(int))
    medias = [
        np.mean([dist for dist, _ in filtrar_vecinos_knn(dist, idx)])
        for dist, idx in zip(distancias, indices)
    ]
    return np.array(medias)


def configurar_eje_x(ax, df_train, tiempo):
    if tiempo == "dias":
        dias = sorted(df_train[tiempo].unique())
        fechas = [FECHA_BASE + pd.Timedelta(days=int(dia)) for dia in dias]
        ax.set_xlim(min(dias) - 2, max(dias) + 2)
        ax.set_xticks(dias)
        ax.set_xticklabels([fecha.strftime("%d-%m") for fecha in fechas])
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        horas = sorted(df_train[(df_train["horas"] >= 0) & (df_train["horas"] <= 23)]["horas"].unique())
        ax.set_xticks(horas)
        ax.set_xlim(-0.75, 23.75)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def graficar_resultados(df_train, df_prueba, var, tiempo, metodo, k, umbral, y_true, y_pred):
    x_train = matriz_puntos(df_train, var, tiempo)
    xx, yy, grid_points = crear_malla(df_train, var, tiempo)
    dist_media_grid = calcular_distancia_malla(x_train, grid_points, metodo, k).reshape(xx.shape)

    x = df_prueba[tiempo]
    y = df_prueba[var]
    mascaras = {
        "TN (normales bien detectados)": (y_true == 0) & (y_pred == 0),
        "FP (normales marcados como anomalos)": (y_true == 0) & (y_pred == 1),
        "TP (anomalos bien detectados)": (y_true == 1) & (y_pred == 1),
        "FN (anomalos marcados como normales)": (y_true == 1) & (y_pred == 0),
    }

    grupos = (
        ("TN y FP", ("TN (normales bien detectados)", "FP (normales marcados como anomalos)")),
        ("TP y FN", ("TP (anomalos bien detectados)", "FN (anomalos marcados como normales)")),
    )

    for titulo, etiquetas in grupos:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.scatter(
            df_train[tiempo],
            df_train[var],
            c="blue" if var == "temperatura" else "purple",
            s=50,
            label="Mediciones Correctas",
        )
        ax.contour(xx, yy, dist_media_grid, levels=[umbral], colors="red", linewidths=2)

        for etiqueta in etiquetas:
            es_error = etiqueta.startswith("FP") or etiqueta.startswith("FN")
            ax.scatter(
                x[mascaras[etiqueta]],
                y[mascaras[etiqueta]],
                c="red" if etiqueta.startswith("TP") or etiqueta.startswith("FP") else "green",
                marker="x" if es_error else "o",
                s=80 if es_error else 20,
                label=etiqueta,
            )

        configurar_eje_x(ax, df_train, tiempo)
        ax.set_xlabel("Fecha" if tiempo == "dias" else "Hora", fontweight="bold")
        ax.set_ylabel("Temperatura (C)" if var == "temperatura" else "Humedad (%)", fontweight="bold")
        ax.set_title(f"{titulo} - {var.capitalize()} por {tiempo} ({metodo}) ({k} vecinos)", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="red", linewidth=2, label="Umbral KNN"))
        labels.append("Umbral KNN")
        ax.legend(handles, labels, loc="upper left")
        fig.tight_layout()

        if GUARDAR_GRAFICAS:
            GRAFICAS_DIR.mkdir(exist_ok=True)
            nombre = f"{titulo}_{var}_{tiempo}_{metodo}_{k}_vecinos.png"
            nombre = nombre.replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(GRAFICAS_DIR / nombre, dpi=300, bbox_inches="tight")

        if MOSTRAR_GRAFICAS:
            plt.show()
        else:
            plt.close(fig)


def evaluar(df_train, df_prueba, var, tiempo, y_pred):
    columna_etiqueta = f"anomalia_{var}_{tiempo}"
    if columna_etiqueta not in df_prueba.columns:
        raise ValueError(
            f"El dataset de prueba debe incluir la columna '{columna_etiqueta}' "
            f"para evaluar {var}."
        )

    y_true = df_prueba[columna_etiqueta].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return y_true, {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def detectar_anomalias(df_train, df_prueba, var, tiempo, metodo, k):
    print(f"\n{var.capitalize()} por {tiempo.capitalize()} - Metodo: {metodo} - Vecinos: {k}")

    x_train = matriz_puntos(df_train, var, tiempo)
    x_prueba = matriz_puntos(df_prueba, var, tiempo)
    tiempos = []
    for _ in range(REPETICIONES_TIEMPO):
        inicio = perf_counter()
        umbral, distancias, vecinos, anomalias = entrenar_y_predecir(x_train, x_prueba, metodo, k)
        tiempos.append(perf_counter() - inicio)
    tiempo_ejecucion = float(np.median(tiempos))

    if MOSTRAR_MUESTRA:
        imprimir_muestra(var, tiempo, x_train, x_prueba, vecinos, distancias, umbral, anomalias)

    y_pred = anomalias.astype(int)
    y_true, metricas = evaluar(df_train, df_prueba, var, tiempo, y_pred)
    graficar_resultados(df_train, df_prueba, var, tiempo, metodo, k, umbral, y_true, y_pred)
    print(f"Tiempo de ejecucion: {tiempo_ejecucion:.4f} s")

    return {
        "Metodo": f"{var.capitalize()} por {tiempo} ({metodo}) ({k} vecinos)",
        "Tiempo_s": round(tiempo_ejecucion, 4),
        **metricas,
    }


def main():
    inicio_total = perf_counter()
    ruta_entrenamiento = DATOS_DIR / "ft.csv"
    if ruta_entrenamiento.exists():
        print(f"Cargando entrenamiento desde CSV: {ruta_entrenamiento}")
        df_entrenamiento = cargar_entrenamiento_csv(ruta_entrenamiento)
    else:
        print("Cargando entrenamiento desde PostgreSQL")
        engine = crear_engine()
        df_entrenamiento = cargar_entrenamiento(engine)

    if "--ejemplo-knn" in sys.argv:
        imprimir_ejemplo_consola(df_entrenamiento)
        return

    df_prueba = cargar_prueba(DATOS_DIR / "datasetSintetico.csv")

    resultados = []
    for metodo in METODOS:
        for k in N_VECINOS:
            for var, tiempo in EXPERIMENTOS:
                df_train = dataset_entrenamiento(df_entrenamiento, var, tiempo)
                df_eval = dataset_prueba(df_prueba, var, tiempo)
                resultados.append(detectar_anomalias(df_train, df_eval, var, tiempo, metodo, k))

    df_resultados = pd.DataFrame(resultados)
    columnas = ["Metodo", "Tiempo_s", "TP", "FP", "TN", "FN", "Precision", "Recall", "Accuracy", "F1-Score"]
    df_resultados = df_resultados[columnas]
    df_resultados.to_csv(DATOS_DIR / "resultados_knn.csv", index=False)
    print(df_resultados)
    print(f"\nTiempo total KNN: {perf_counter() - inicio_total:.4f} s")


if __name__ == "__main__":
    main()
