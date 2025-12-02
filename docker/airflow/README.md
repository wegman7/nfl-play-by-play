# Airflow Setup

This directory contains the Docker-based Airflow setup for the NFL play-by-play prediction project.

## Directory Structure

```
docker/airflow/
├── docker-compose.yaml   # Airflow services configuration
├── .env                  # Environment variables
├── dags/                 # Airflow DAGs (symlinked or copied from ../../dags/)
├── logs/                 # Airflow logs (auto-generated)
└── plugins/              # Custom Airflow plugins
```

## Quick Start

### 1. Initialize Airflow

From the `docker/airflow/` directory:

```bash
cd docker/airflow
docker compose up airflow-init
```

This will:
- Create necessary directories
- Install Python dependencies from `requirements.txt`
- Initialize the Airflow database
- Create the admin user

### 2. Start Airflow

```bash
docker compose up -d
```

This starts:
- PostgreSQL database (for Airflow metadata)
- Airflow webserver (http://localhost:8081)
- Airflow scheduler

### 3. Access Airflow UI

Open http://localhost:8081 in your browser.

**Default credentials:**
- Username: `airflow`
- Password: `airflow`

### 4. Stop Airflow

```bash
docker compose down
```

To remove volumes (reset database):
```bash
docker compose down -v
```

## Configuration

### Environment Variables

Edit [.env](.env) to configure:
- `AIRFLOW_UID`: Your system user ID (default: 501)
- `_AIRFLOW_WWW_USER_USERNAME`: Webserver username
- `_AIRFLOW_WWW_USER_PASSWORD`: Webserver password
- `_PIP_ADDITIONAL_REQUIREMENTS`: Extra pip packages

### Volumes

The docker-compose.yaml mounts:
- `dags/` → `/opt/airflow/dags` (DAG definitions)
- `logs/` → `/opt/airflow/logs` (execution logs)
- `plugins/` → `/opt/airflow/plugins` (custom plugins)
- `../../src` → `/opt/airflow/src` (project source code)
- `../../data` → `/opt/airflow/data` (data files)
- `../../mlflow.db` → `/opt/airflow/mlflow.db` (MLflow tracking)
- `../../requirements.txt` → installed at init

## Adding DAGs

Place DAG files in `docker/airflow/dags/`. They will be automatically detected by Airflow.

Example:
```bash
# Copy existing DAGs
cp ../../dags/*.py docker/airflow/dags/

# Or create a symlink
ln -s ../../dags docker/airflow/dags
```

## Troubleshooting

### Permission errors

If you see permission errors, update `AIRFLOW_UID` in `.env` to match your user ID:
```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

### Dependencies not installing

Ensure `requirements.txt` exists in the project root. Add any missing packages there.

### DAGs not appearing

- Check that DAG files are in `dags/` directory
- Verify no Python syntax errors in DAG files
- Check logs: `docker compose logs airflow-scheduler`

### Reset everything

```bash
docker compose down -v
rm -rf logs/*
docker compose up airflow-init
docker compose up -d
```

## Integration with Kafka

To connect Airflow with the existing Kafka setup (in `docker-compose.yaml` at project root), you can use Docker networks or reference `localhost:19092` from within Airflow containers.

## Ports

- **8081**: Airflow webserver (note: Kafka UI uses 8080)
- **5432**: PostgreSQL (internal only)
