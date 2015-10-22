def fourcc((a, b, c, d)):
    """
    Calculate a FourCC integer from the four characters.

    http://www.fourcc.org/

    """

    return (((((ord(d) << 8) | ord(c)) << 8) | ord(b)) << 8) | ord(a)
