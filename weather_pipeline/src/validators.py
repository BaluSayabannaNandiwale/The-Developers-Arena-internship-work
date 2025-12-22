def validate_weather(data):
    if data is None:
        return False

    if "temperature" not in data or "humidity" not in data or "pressure" not in data:
        return False

    if not (-50 <= data["temperature"] <= 60):
        return False

    if not (0 <= data["humidity"] <= 100):
        return False

    if data["pressure"] <= 0:
        return False

    return True
