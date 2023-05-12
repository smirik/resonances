def convert_mjd_to_date(mjd: float) -> str:
    from datetime import datetime, timedelta

    base_date = datetime(1858, 11, 17)
    delta = timedelta(days=mjd)
    date = base_date + delta
    formatted_date = date.strftime("%Y-%m-%d")
    return formatted_date
