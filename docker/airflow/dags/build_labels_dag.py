from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from play_by_play.pipelines.build_labels import run_build_labels


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="build_labels_dag",
    default_args=default_args,
    description="Build NFL play-by-play labels from raw data",
    schedule=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["nfl", "labels", "preprocessing"],
) as dag:

    build_labels_task = PythonOperator(
        task_id="run_build_labels",
        python_callable=run_build_labels,
    )
