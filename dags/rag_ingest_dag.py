from datetime import datetime
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

# These imports work when the repo folder is mounted into the Airflow image
import sys
REPO_ROOT = "/opt/airflow/repo"    # mount your project here via docker-compose
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from pipeline.build_index import run_build_index

default_args = {
    "owner": "rag-poc",
    "retries": 0,
}

with DAG(
    dag_id="rag_ingest_dag",
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,   # trigger manually for the PoC
    catchup=False,
    default_args=default_args,
    tags=["rag","qdrant","e5"],
) as dag:

    build_index = PythonOperator(
        task_id="build_index",
        python_callable=run_build_index,
    )

    build_index
