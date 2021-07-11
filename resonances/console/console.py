import argparse
from pathlib import Path


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls


@static_init
class console:
    args = None
    parser = None

    @classmethod
    def static_init(cls):
        cls.parser = argparse.ArgumentParser(description='')

    @classmethod
    def create_output_dir(cls, output):
        Path(output).mkdir(parents=True, exist_ok=True)

    @classmethod
    def shape(cls):
        cls.parser.add_argument('--config', nargs='?', type=str)
        cls.args = cls.parser.parse_args()

    @classmethod
    def asteroid(cls):
        cls.parser.add_argument('--config', nargs='?', type=str)
        cls.args = cls.parser.parse_args()

    @classmethod
    def quick(cls):
        cls.parser.add_argument('asteroid', help='The number of the asteroid you want to research.', type=int)
        cls.parser.add_argument('-r', '--resonance', nargs='+', help='<Required> Setup the resonance', required=True, type=int)
        cls.args = cls.parser.parse_args()


# def test_static_init():
#     assert SomeEnum.text_dict["Val A"] == SomeEnum.VAL_A
#     assert SomeEnum.text_dict["Val B"] == SomeEnum.VAL_B
#     assert SomeEnum.text_dict["Val C"] == SomeEnum.VAL_C
#     assert SomeEnum.text_dict["Val D"] == SomeEnum.VAL_D
