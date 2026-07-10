@echo off
cd docker
docker compose --env-file ../.env up --build -d