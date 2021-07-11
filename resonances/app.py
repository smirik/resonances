import time

from resonances.simulation import shape
from resonances.resonance.three_body import ThreeBody

start_time = time.time()


def main():
    elem = {
        "m": 0.0,
        "a": 2.398825840331548,
        "e": 0.2194125828625336,
        "inc": 0.23627318991620527,
        "Omega": 0.6370508455573044,
        "omega": 5.752902062786396,
        "M": 2.4309844848211464,
        "label": "A463",
    }
    variations = {"a": {"min": 2.388825840331548, "max": 2.408825840331548, "num": 3}, "e": {"min": 0.0, "max": 0.3, "num": 3}}
    mmr_template = ThreeBody([4.0, -2.0, -1.0, 0.0, 0.0, -1.0], [5, 6], 10, '{}'.format(elem['label']))
    shape.run(
        elem,
        variations,
        mmr_template,
        save=True,
        save_path="cache/simulation-shape",
        need_plot=True,
        dump=3,
    )


def run():
    a = 5.0
    print("Starting program")
    main()
    print("Ending program")
    print("--- %s seconds ---" % (time.time() - start_time))
    # test.my.hello()


def version():
    return "0.1.0"
