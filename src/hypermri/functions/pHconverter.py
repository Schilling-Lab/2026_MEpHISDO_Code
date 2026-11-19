import numpy as np

# Larmor Frequencies (MHz) at 7 Tesla from here:
# https://kodu.ut.ee/~laurit/AK2/NMR_tables_Bruker2012.pdf
LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468
LAMOR_FREQUENCY_P31 = 121.495

# [pK, δHA, δA = δHA - δ_range] - Reference: Urea
PARAMS_OMPD_C1 = 6.5450, 8.4919, 8.4919 - 0.4644

# [pK, δHA, δA = δHA + δ_range] - Reference: Urea
PARAMS_OMPD_C5 = 6.5450, 12.4801, 12.4801 + 3.2975

# [pK, δHA, δA]
PARAMS_PHOSPHOR = [6.77, 3.23, 5.70]


def pH2ppm(ph_value, params):
    """Reversed Henderson-Hasselbach relationship to convert pH to chemical shift."""
    pK, dHA, dA = params
    return dHA + (dA - dHA) / (1 + 10 ** (-(ph_value - pK)))


def ppm2pH(ppm, params):
    """Henderson-Hasselbach relationship to convert chemical shift to pH."""
    pK, dHA, dA = params
    return pK + np.log10((ppm - dHA) / (dA - ppm))


def freq2pH(frequency, params, lamor_frequency):
    """Henderson-Hasselbach relationship to convert chemical shift (given in Hz) to pH."""
    return ppm2pH(frequency / lamor_frequency, params)


def pH2freq(ppm, params, lamor_frequency):
    """Henderson-Hasselbach relationship to convert pH to chemical shift (given in Hz)."""
    return ppm2pH(ppm, params) * lamor_frequency


def OMPD5_freq2pH(freq, params=PARAMS_OMPD_C5, lamor_frequency=LAMOR_FREQUENCY_C13):
    """Convert frequency of OMPD C5 peak into pH."""
    return freq2pH(freq, params, lamor_frequency)
