import os


class Config:
    DEBUG = (
        os.getenv("DEBUG", "False") == "1"
        or os.getenv("DEBUG", "True").lower() == "true"
    )
    CACHING = (
        os.getenv("CACHING", "False") == "1"
        or os.getenv("CACHING", "True").lower() == "true"
    )
    TESTING = False


class TestingConfig:
    DEBUG = (
        os.getenv("DEBUG", "False") == "1"
        or os.getenv("DEBUG", "True").lower() == "true"
    )
    CACHING = (
        os.getenv("CACHING", "False") == "1"
        or os.getenv("CACHING", "True").lower() == "true"
    )
    TESTING = True
