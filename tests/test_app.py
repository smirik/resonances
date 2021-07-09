from resonances import __version__
from resonances import app

def test_version():
    assert __version__ == '0.1.0'

def test_app_version():
    assert app.version() == '0.1.0'