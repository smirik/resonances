import argparse


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls


@static_init
class console:
    args = None

    @classmethod
    def static_init(cls):
        parser = argparse.ArgumentParser(description='Identify if an asteroid has a given resonance.')
        parser.add_argument('--config', nargs='?', type=str)
        cls.args = parser.parse_args()


# def test_static_init():
#     assert SomeEnum.text_dict["Val A"] == SomeEnum.VAL_A
#     assert SomeEnum.text_dict["Val B"] == SomeEnum.VAL_B
#     assert SomeEnum.text_dict["Val C"] == SomeEnum.VAL_C
#     assert SomeEnum.text_dict["Val D"] == SomeEnum.VAL_D
