from scripts.verify_megascale_quotes import QUOTES, verify


def test_quote_verifier_requires_all_exact_sentences():
    text = "\n".join(QUOTES.values())
    assert all(verify(text).values())
    partial = "\n".join(list(QUOTES.values())[:-1])
    results = verify(partial)
    assert sum(results.values()) == len(QUOTES) - 1
