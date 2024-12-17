import os


class Config:
    DEBUG = os.getenv("DEBUG", True)
    TESTING = False


class TestingConfig:
    DEBUG = os.getenv("DEBUG", True)
    TESTING = True
