from src.database import setup_database, get_or_create_city

def test_city_insert():
    setup_database()
    city_id = get_or_create_city("TestCity")
    assert city_id is not None
