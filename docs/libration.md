# The workflow of identification

1. If there is a pure libration around 0 or Pi, then the function ends with the status `2`. Otherwise:
2. If there is no peaks on [the Lomb-Scargle periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html), then the function exists with the status `1'.
3. If there are peaks, it might be the case that they are false positives (i.e. long-term circulation). In this case, the app performs [a kernel-density estimation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html). The density is calculated over the values of the resonant angle (from 0 to 2\*pi).
4. If there are peaks on this diagram, it means that there is a libration not a circulation because for the latter, one might expect to see almost horizontal line. Therefore, the function exists with the status `1`. Otherwise --- with `0'.

## Possible statuses

| Status | Description                                                                         |
| :----: | ----------------------------------------------------------------------------------- |
|   0    | There is no libration (circulation or similar).                                     |
|   1    | There is a libration but it does not look pure (more specifically, around 0 or Pi). |
|   2    | There is a pure libration around 0 or Pi.                                           |
