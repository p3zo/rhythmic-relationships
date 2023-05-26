from rhythmic_relationships.data import PAD_IX, get_pair_sequences


def test_get_pair_sequences():
    p1 = [1, 3, 4]
    p2 = [1, 2, 2]
    context_len = 3
    X, Y = get_pair_sequences(p1, p2, context_len)
    assert X == [[PAD_IX, PAD_IX, 1], [PAD_IX, 1, 3], [1, 3, 4]]
    assert Y == [[PAD_IX, PAD_IX, 1], [PAD_IX, 1, 2], [1, 2, 2]]
