import os
from contextlib import contextmanager

import psycopg2


def _get_discrete_config() -> dict:
	return {
		"host": os.getenv("POSTGRES_HOST", "localhost"),
		"port": int(os.getenv("POSTGRES_PORT", "5432")),
		"dbname": os.getenv("POSTGRES_DB", "ecostream"),
		"user": os.getenv("POSTGRES_USER", "postgres"),
		"password": os.getenv("POSTGRES_PASSWORD", "postgres"),
	}


def get_connection():
	postgres_url = os.getenv("POSTGRES_URL")
	if postgres_url:
		return psycopg2.connect(postgres_url)
	return psycopg2.connect(**_get_discrete_config())


@contextmanager
def db_connection():
	conn = get_connection()
	try:
		yield conn
	finally:
		conn.close()
