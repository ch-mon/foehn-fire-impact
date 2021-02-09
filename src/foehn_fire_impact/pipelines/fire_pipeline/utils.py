def decimalWSG84_to_LV3(lon, lat):
    '''
    Convert WSG84 to LV3 coordinates
    :param lon:
    :param lat:
    :return:
    '''
    phi = lat * 3600
    lambda_ = lon * 3600

    phi = (phi - 169028.66) / 10000
    lambda_ = (lambda_ - 26782.5) / 10000

    E = 2600072.37 + 211455.93 * lambda_ - 10938.51 * lambda_ * phi - 0.36 * lambda_ * phi * phi - 44.54 * lambda_ * lambda_ * lambda_
    y = E - 2000000.0

    N = 1200147.07 + 308807.95 * phi + 3745.25 * lambda_ * lambda_ + 76.63 * phi * phi - 194.56 * lambda_ * lambda_ * phi + 119.79 * phi * phi * phi
    x = N - 1000000.0

    return x, y


def LV3_to_decimalWSG84(x, y):
    '''
    Convert LV3 to WSG84 coordinates
    :param x:
    :param y:
    :return:
    '''
    y = (y - 600000) / 1000000
    x = (x - 200000) / 1000000

    lambda_ = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x * x - 0.0436 * y * y * y
    phi = 16.9023892 + 3.238272 * x - 0.270978 * y * y - 0.002528 * x * x - 0.0447 * y * y * x - 0.0140 * x * x * x

    lon = lambda_ * 100 / 36
    lat = phi * 100 / 36

    return lon, lat