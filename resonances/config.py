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
    def static_init(cls):
        config_file_path = 'config.json'
        config_file = Path(config_file_path)

        if not config_file.exists():
            raise Exception('No config.json presented. Cannot continue working.')

        with open('config.json', "r") as read_file:
            cls.config = json.load(read_file)
