def _binomial(n : int, m: int):
    if m == 0:
        return 1
    elif m > n / 2:
        return _binomial(n, n - m)
    else:
        return (n / m) * _binomial(n - 1, m - 1)

def binomial(n : int, m: int):
    if not isinstance(n, int):
        raise ValueError("n must be integer.")
    if not isinstance(m, int):
        raise ValueError("m must be integer.")
    if m < 0 or m > n:
        raise ValueError("m must be between 0 and n.")
    return _binomial(n, m)
