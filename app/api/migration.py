from sqlalchemy import create_engine
import os
import pandas as pd

DATABASE_URL = os.environ.get("DATABASE_URL")



def migrate_csv_to_postgresql():

    CSV_FILE_PATH = "data/voter_intentions_COMPLETED.csv"
    TABLE_NAME = "datos" 

    try:
        print(f"Leyendo el archivo CSV: {CSV_FILE_PATH}...")
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Se cargaron {len(df)} filas.")

        engine = create_engine(f"{DATABASE_URL}?sslmode=require")

        print(f"Conectando a la base de datos y migrando datos a la tabla '{TABLE_NAME}'...")

        df.to_sql(
            TABLE_NAME,
            engine,
            if_exists='replace',
            index=False 
        )

        print("\n¡Migración completada exitosamente!")
        print(f"Tu tabla '{TABLE_NAME}' ahora está en PostgreSQL.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en '{CSV_FILE_PATH}'")
    except Exception as e:
        print(f"Ocurrió un error durante la migración:")

def get_csv_from_postgresql():
    TABLE_NAME = "datos"  
    dataframe = pd.DataFrame()

    try:
        engine = create_engine(f"{DATABASE_URL}?sslmode=require")

        print(f"Conectando a la base de datos y leyendo datos de la tabla '{TABLE_NAME}'...")

        df = pd.read_sql_table(TABLE_NAME, engine)

        print(f"Se leyeron {len(df)} filas.")

        dataframe = df

    except Exception as e:
        print(f"Ocurrió un error durante la lectura:")

    print("\n¡Lectura completada exitosamente!")

    return dataframe


def add_value_to_postgresql(new_data: dict):
    TABLE_NAME = "datos"  
    
    print("Agregando nueva fila a PostgreSQL...")

    try:
        engine = create_engine(f"{DATABASE_URL}?sslmode=require")

        print(f"Conectando a la base de datos y agregando nueva fila a la tabla '{TABLE_NAME}'...")

        df_new = pd.DataFrame([new_data])

        df_new.to_sql(
            TABLE_NAME,
            engine,
            if_exists='append',
            index=False 
        )

        print("\n¡Fila agregada exitosamente!")

    except Exception as e:
        print(f"Ocurrió un error durante la inserción:")
        print(e)


def consultar_datos_postgresql():
    TABLE_NAME = "datos"  

    try:
        engine = create_engine(f"{DATABASE_URL}?sslmode=require")

        print(f"Conectando a la base de datos y consultando datos de la tabla '{TABLE_NAME}'...")

        df = pd.read_sql_table(TABLE_NAME, engine)

        print(f"Se leyeron {len(df)} filas.")
        print(df.head())

    except Exception as e:
        print(f"Ocurrió un error durante la consulta:")
        print(e)

    return df