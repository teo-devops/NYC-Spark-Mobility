-- config/db/init.sql
-- Se ejecuta automáticamente al primer arranque de MariaDB.
-- Crea la DB de MLflow separada de la de Hive Metastore.

CREATE DATABASE IF NOT EXISTS hive_metastore_mlflow
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

GRANT ALL PRIVILEGES ON hive_metastore_mlflow.* TO 'hive'@'%';
FLUSH PRIVILEGES;
