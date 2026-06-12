import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "Datos"


def cargar_env_privado(ruta):
    if not ruta.exists():
        return

    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue

        clave, valor = linea.split("=", 1)
        clave = clave.strip()
        valor = valor.strip().strip('"').strip("'")
        os.environ.setdefault(clave, valor)


cargar_env_privado(BASE_DIR / ".env")


def get_db_config():
    config = {
        "host": os.getenv("TFG_DB_HOST", "localhost"),
        "database": os.getenv("TFG_DB_NAME"),
        "user": os.getenv("TFG_DB_USER"),
        "password": os.getenv("TFG_DB_PASSWORD"),
        "port": os.getenv("TFG_DB_PORT", "5432"),
    }

    campos_obligatorios = ["database", "user", "password"]
    faltantes = [campo for campo in campos_obligatorios if not config[campo]]
    if faltantes:
        raise RuntimeError(
            "Faltan credenciales de base de datos. Define TFG_DB_NAME, "
            "TFG_DB_USER y TFG_DB_PASSWORD en Codigo/.env o como variables "
            "de entorno."
        )

    return config


def get_db_uri():
    config = get_db_config()
    return (
        f"postgresql+psycopg2://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )
