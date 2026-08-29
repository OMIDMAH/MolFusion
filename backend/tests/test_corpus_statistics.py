import pytest

from molfusion_backend.corpus.statistics import CorpusStatisticsAccumulator, RecordCounts


# ---------------------------------------------------------------------------
# CorpusStatisticsAccumulator
# ---------------------------------------------------------------------------


def test_empty_accumulator_reports_nulls_not_zeros():
    """A corpus with no documents has no minimum length; reporting 0 would
    be a false measurement."""
    report = CorpusStatisticsAccumulator().as_report()

    assert report["document_count"] == 0
    assert report["smiles_length"] == {"min": None, "max": None, "mean": None, "median": None}
    assert report["token_count"]["min"] is None


def test_summarizes_smiles_length():
    accumulator = CorpusStatisticsAccumulator()
    for smiles in ("CCO", "CCCC", "CCCCC"):
        accumulator.add(smiles, token_count=len(smiles))

    lengths = accumulator.as_report()["smiles_length"]
    assert lengths["min"] == 3
    assert lengths["max"] == 5
    assert lengths["median"] == 4
    assert lengths["mean"] == 4.0


def test_summarizes_token_counts_independently_of_length():
    accumulator = CorpusStatisticsAccumulator()
    accumulator.add("[13CH3]CO", token_count=3)
    accumulator.add("CCO", token_count=3)

    report = accumulator.as_report()
    assert report["token_count"] == {"min": 3, "max": 3, "mean": 3.0, "median": 3}
    assert report["smiles_length"]["max"] == 9


def test_median_of_an_even_sample():
    accumulator = CorpusStatisticsAccumulator()
    for length in (1, 2, 3, 4):
        accumulator.add("C" * length, token_count=length)

    assert accumulator.as_report()["smiles_length"]["median"] == 2.5


def test_counts_disconnected_components():
    accumulator = CorpusStatisticsAccumulator()
    accumulator.add("CC(=O)[O-].[Na+]", token_count=9)
    accumulator.add("CCO", token_count=3)

    assert accumulator.as_report()["with_disconnected_components"] == 1


def test_counts_stereochemistry_from_both_marker_families():
    """Tetrahedral chirality shows up as "@"; double-bond stereo as "/" or
    "\\". Both must be recognized, and neither counted twice."""
    accumulator = CorpusStatisticsAccumulator()
    accumulator.add("C[C@H](O)Cl", token_count=7)
    accumulator.add("C/C=C/C", token_count=7)
    accumulator.add("C/C=C\\C", token_count=7)
    accumulator.add("CCO", token_count=3)

    assert accumulator.as_report()["with_stereochemistry"] == 3


def test_a_document_with_both_stereo_and_components_counts_once_in_each():
    accumulator = CorpusStatisticsAccumulator()
    accumulator.add("C[C@H](O)Cl.[Na+]", token_count=11)

    report = accumulator.as_report()
    assert report["with_disconnected_components"] == 1
    assert report["with_stereochemistry"] == 1


def test_accumulating_the_same_documents_twice_gives_identical_reports():
    def build():
        accumulator = CorpusStatisticsAccumulator()
        for index, smiles in enumerate(("CCO", "c1ccccc1", "CC(=O)[O-].[Na+]")):
            accumulator.add(smiles, token_count=index + 1)
        return accumulator.as_report()

    assert build() == build()


def test_report_contains_no_vocabulary_analysis():
    """Phase 5F-B reports corpus shape only -- n-grams and document
    frequencies are Phase 5F-C."""
    accumulator = CorpusStatisticsAccumulator()
    accumulator.add("CCO", token_count=3)

    keys = set(accumulator.as_report())
    assert keys == {
        "document_count",
        "smiles_length",
        "token_count",
        "with_disconnected_components",
        "with_stereochemistry",
    }


# ---------------------------------------------------------------------------
# RecordCounts
# ---------------------------------------------------------------------------


def _balanced_counts() -> RecordCounts:
    return RecordCounts(
        rows_examined=10,
        null_smiles=1,
        empty_smiles=1,
        rdkit_parse_failures=1,
        zero_atom_molecules=1,
        tokenization_failures=0,
        valid_pre_dedup=6,
        duplicate_canonical_smiles=2,
        unique_canonical_smiles=4,
        document_count=4,
    )


def test_balanced_counts_validate():
    _balanced_counts().validate()


def test_a_lost_record_is_caught():
    counts = _balanced_counts()
    counts.valid_pre_dedup = 5  # one record now belongs to no category

    with pytest.raises(ValueError, match="Record accounting does not balance"):
        counts.validate()


def test_a_deduplication_mismatch_is_caught():
    counts = _balanced_counts()
    counts.duplicate_canonical_smiles = 1

    with pytest.raises(ValueError, match="Deduplication accounting"):
        counts.validate()


def test_a_document_count_mismatch_is_caught():
    counts = _balanced_counts()
    counts.document_count = 3

    with pytest.raises(ValueError, match="Document accounting"):
        counts.validate()


def test_tokenization_failures_are_an_exclusion_category():
    """A record dropped for violating the tokenizer contract is excluded
    before deduplication, so it reduces valid_pre_dedup rather than the
    document count."""
    counts = _balanced_counts()
    counts.tokenization_failures = 1
    counts.valid_pre_dedup = 5
    counts.unique_canonical_smiles = 3
    counts.document_count = 3

    counts.validate()


def test_an_uncounted_tokenization_failure_is_caught():
    counts = _balanced_counts()
    counts.valid_pre_dedup = 5  # a record left every category

    with pytest.raises(ValueError, match="Record accounting does not balance"):
        counts.validate()


def test_counts_serialize_every_documented_category():
    report = _balanced_counts().as_report()
    assert set(report) == {
        "rows_examined",
        "null_smiles",
        "empty_smiles",
        "rdkit_parse_failures",
        "zero_atom_molecules",
        "valid_pre_dedup",
        "duplicate_canonical_smiles",
        "unique_canonical_smiles",
        "tokenization_failures",
        "document_count",
    }
