import logging
import resonances.config


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls


@static_init
class logger:
    @classmethod
    def static_init(cls):
        logging.basicConfig(
            filename=resonances.config.get('log.file'),
            encoding='utf-8',
            level=cls.get_logging_level(),
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    @classmethod
    def get_logging_level(cls):
        config_level = resonances.config.get('log.level')
        if 'debug' == config_level:
            return logging.DEBUG
        elif 'warning' == config_level:
            return logging.WARNING
        elif 'error' == config_level:
            return logging.ERROR
        elif 'critical' == config_level:
            return logging.CRITICAL

        return logging.INFO

    @classmethod
    def info(cls, message):
        logging.info(message)

    @classmethod
    def debug(cls, message):
        logging.debug(message)

    @classmethod
    def warning(cls, message):
        logging.warning(message)

    @classmethod
    def error(cls, message):
        logging.error(message)
