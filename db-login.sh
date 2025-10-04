#!/bin/bash
# Login to reggie PostgreSQL database

PGPASSWORD=postgres psql -h localhost -p 5432 -U johntermaat -d reggie
