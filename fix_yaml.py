with open("docker-compose.yml", "r") as f:
    lines = f.readlines()
with open("docker-compose.yml", "w") as f:
    for line in lines:
        if line.startswith("      - ./mlflow_data:/mlflow"):
            # We already added the bad lines maybe? Let's check how many times it appears in jupyter
            pass
