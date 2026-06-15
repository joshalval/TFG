# Deteccion de anomalias en datos IoT

Codigo desarrollado para el Trabajo de Fin de Grado de Jose Haldón centrado en la deteccion de anomalias en mediciones de temperatura y humedad obtenidas mediante sensores IoT de bajo coste.

El proyecto compara dos enfoques:

- K-Nearest Neighbours no supervisado.
- Entropia de Shannon como metodo estadistico.

## Estructura del repositorio

```text
Datos/                 CSV de entrada y resultados numericos
Graficas/              Graficas descriptivas de los datos
Graficas_KNN/          Graficas generadas por el metodo KNN
Graficas_Entropia/     Graficas generadas por entropia de Shannon
config.py              Configuracion comun y carga opcional de base de datos
graficas.py            Generacion de graficas descriptivas
datasetSintetico.py    Generacion del conjunto sintetico de evaluacion
KNN.py                 Deteccion de anomalias mediante KNN
entropia.py            Deteccion de anomalias mediante entropia de Shannon
requirements.txt       Dependencias de Python
```

## Datos incluidos

La carpeta `Datos` contiene los CSV necesarios para ejecutar el proyecto sin depender de una base de datos local:

- `ft.csv`: mediciones originales exportadas.
- `ft_agregada_dias.csv`: mediciones agregadas por dias.
- `ft_agregada_horas.csv`: mediciones agregadas por horas.
- `datasetSintetico.csv`: conjunto de evaluacion generado a partir de los datos reales.
- `resultados_knn.csv`: resultados finales del metodo KNN.
- `resultados_entropia.csv`: resultados finales del metodo de entropia.

## Instalacion

Se recomienda Python 3.11.

## Ejecucion

Ejecutar los scripts desde la carpeta raiz del repositorio:

```powershell
python graficas.py
python datasetSintetico.py
python KNN.py
python entropia.py
```

Orden recomendado:

1. `graficas.py`: genera las graficas descriptivas de temperatura y humedad.
2. `datasetSintetico.py`: genera `Datos/datasetSintetico.csv`.
3. `KNN.py`: ejecuta el metodo KNN y actualiza `Datos/resultados_knn.csv`.
4. `entropia.py`: ejecuta el metodo de entropia de Shannon y actualiza `Datos/resultados_entropia.csv`.

## Salidas principales

- `Graficas/`: visualizacion descriptiva de los datos.
- `Graficas_KNN/`: resultados visuales del algoritmo KNN.
- `Graficas_Entropia/`: resultados visuales del metodo de entropia.
- `Datos/resultados_knn.csv`: metricas finales de KNN.
- `Datos/resultados_entropia.csv`: metricas finales de entropia.

## Configuracion opcional de base de datos

Los scripts usan primero los CSV incluidos en `Datos`. Si esos archivos no existen, algunos scripts pueden leer datos desde PostgreSQL mediante variables de entorno:

```text
TFG_DB_HOST
TFG_DB_PORT
TFG_DB_NAME
TFG_DB_USER
TFG_DB_PASSWORD
```

Tambien se puede crear un archivo local `.env` con esas variables. Este archivo no debe subirse al repositorio.

## Notas

- Las graficas y CSV de resultados incluidos corresponden a la ejecucion final usada en la memoria del TFG.
