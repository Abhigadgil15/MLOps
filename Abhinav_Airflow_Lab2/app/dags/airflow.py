from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow import configuration as conf
from datetime import datetime, timedelta

# Import your functions from lab.py
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Enable pickle-based XComs (for passing serialized data)
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments for the DAG
default_args = {
    'owner': 'abhinav',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 10, 6),
}

# Define the DAG
with DAG(
    dag_id='regression_pipeline_dag',
    default_args=default_args,
    description='A DAG for Linear Regression data processing and model training',
    schedule_interval='0 12 * * *',  # runs every day at 12:00 PM
    catchup=False,
    tags=['regression', 'lab1']
) as dag:

    # Start message
    start_task = BashOperator(
        task_id='start_message',
        bash_command='echo "Starting the Airflow Regression DAG pipeline..."'
    )

    # Task 1: Load data
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data
    )

    # Task 2: Preprocess data
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output]
    )

    # Task 3: Build and save regression model
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, 'regression_model.pkl']
    )

    # Task 4: Load model and predict on test data
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=['regression_model.pkl', build_save_model_task.output]
    )

    # End message
    end_task = BashOperator(
        task_id='end_message',
        bash_command='echo "Regression pipeline completed successfully!"'
    )

    # Define task dependencies
    start_task >> load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> end_task
