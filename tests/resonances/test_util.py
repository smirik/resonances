from resonances.util import convert_mjd_to_date


def test_convert_mjd_to_date():
    assert '2023-02-25' == convert_mjd_to_date(60000)
    assert '2020-05-31' == convert_mjd_to_date(59000)
