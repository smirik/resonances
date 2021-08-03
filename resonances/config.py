import json
from pathlib import Path


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls


@static_init
class config:
    config = None

    @classmethod
    def get(cls, key):
        try:
            value = cls.config[key]
        except KeyError:
            raise Exception('There is no config with key = {}. The full config: {}'.format(key, json.dumps(cls.config)))
        except Exception:
            raise
        return value

    @classmethod
    def has(cls, key):
        if key in cls.config:
            return True
        return False

    @classmethod
    def set(cls, key, value):
        if not cls.has(key):
            raise Exception('There is no config with key = {}. The full config: {}'.format(key, json.dumps(cls.config)))

        cls.config[key] = value

    @classmethod
    def static_init(cls):
        config_file_dir = Path(__file__).parent.resolve()
        config_file_path = '{}/config.json'.format(str(config_file_dir))
        config_file = Path(config_file_path)

        if not config_file.exists():
            raise Exception('No config.json presented. Looking at {} Cannot continue working.'.format(config_file_path))

        with open(config_file_path, "r") as read_file:
            cls.config = json.load(read_file)
