"""High-order MMRs for docs/examples.md, in a process of its own: the catalogue limits must be set before the
catalogue is first read."""

# --8<-- [start:high_order]
import resonances

# Higher orders than the default catalogues (2-body order <= 10, 3-body order <= 6): set the limits and a new file
# before anything reads the catalogue (the cached one, built with the old limits, would be read as it is)
resonances.config.set('MATRIX_2BODY_PRIMARY_MAX', 40)  # exclusive: the planet's coefficient runs up to 39
resonances.config.set('MATRIX_2BODY_COEF_MAX', 39)
resonances.config.set('MATRIX_2BODY_ORDER_MAX', 25)
resonances.config.set('MATRIX_2BODY_FILE', 'cache/mmr-2body-order25.csv')

high = resonances.find_mmrs(2.3986, planets=['Mars', 'Jupiter'], sigma=0.005, mmr_type='two-body')
print([mmr.to_short() for mmr in high])
# --8<-- [end:high_order]
