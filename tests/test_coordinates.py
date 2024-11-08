from spice_segmenter.coordinates import Vector


def test_vector():
    t1 = "2024-12-25T00:00:00"
    vec = Vector("JUICE_SPACECRAFT", "EARTH", frame="JUICE_SPACECRAFT")

    vec(t1)

    vec_radec = vec.as_radec

    vec_radec(t1)
