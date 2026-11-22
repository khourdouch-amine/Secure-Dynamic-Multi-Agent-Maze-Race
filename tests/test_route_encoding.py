import pytest

from planners.quantum.route_encoding import RouteEncoding


def test_encode_and_decode_round_trip():
    sequence = (0, 3, 1, 0)

    index = RouteEncoding.encode(sequence)

    assert index == 28
    assert RouteEncoding.decode(index, 4) == sequence


def test_all_eight_valid_route_encodings_are_unique():
    sequences = [
        (0, 0, 2, 0),
        (0, 0, 3, 1),
        (0, 2, 0, 0),
        (0, 3, 0, 1),
        (0, 3, 1, 0),
        (3, 0, 0, 1),
        (3, 0, 1, 0),
        (3, 1, 0, 0),
    ]

    indices = [RouteEncoding.encode(sequence) for sequence in sequences]

    assert len(indices) == 8
    assert len(set(indices)) == 8

    for sequence, index in zip(sequences, indices):
        assert RouteEncoding.decode(index, 4) == sequence


def test_decode_qiskit_bitstring():
    assert RouteEncoding.decode_bitstring("00011100", 4) == (0, 3, 1, 0)


def test_sequence_to_path():
    sequence = (0, 3, 1, 0)

    path = RouteEncoding.sequence_to_path((1, 1), sequence)

    assert path == [
        (1, 1),
        (2, 1),
        (2, 2),
        (2, 1),
        (3, 1),
    ]


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError):
        RouteEncoding.encode((0, 4, 1, 0))


def test_invalid_index_is_rejected():
    with pytest.raises(ValueError):
        RouteEncoding.decode(256, 4)


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError):
        RouteEncoding.encode(())
