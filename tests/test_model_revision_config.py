from scripts.check_model_revisions import check_model_revisions


def test_model_and_tokenizer_revisions_are_pinned():
    assert check_model_revisions()