import csv
import json

import pytest

from molfusion_backend.artifacts.checksum import sha256_file
from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
from molfusion_backend.corpus.errors import CorpusIdentityError, CorpusOutputExistsError
from molfusion_backend.corpus.serialization import CORPUS_SERIALIZATION_ID, write_corpus
from molfusion_backend.corpus.study import runner
from molfusion_backend.corpus.study.ngrams import NgramFrequencyAccumulator
from molfusion_backend.corpus.study.report import (
    deterministic_study_view,
    run_study,
    study_report_bytes,
)
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID

# Enough molecules, and enough variety of length, that both split buckets
# are populated and every n-gram order is exercised. Small enough that a
# full study run costs a fraction of a second.
FIXTURE_SMILES = sorted(
    {"CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CC(=O)N", "CCCl", "CCBr", "N", "O"}
    | {f"C{'C' * index}O" for index in range(1, 220)}
    | {f"c1ccccc1{'C' * index}" for index in range(1, 90)}
)


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "canonical_smiles.smi"
    sha256, _ = write_corpus(path, FIXTURE_SMILES)
    return path, sha256


def study(corpus, tmp_path, **kwargs):
    path, sha256 = corpus
    kwargs.setdefault("expected_sha256", sha256)
    kwargs.setdefault("expected_document_count", len(FIXTURE_SMILES))
    kwargs.setdefault("progress_every", 0)
    return run_study(path, tmp_path / kwargs.pop("out", "study"), **kwargs)


# ---------------------------------------------------------------------------
# corpus identity
# ---------------------------------------------------------------------------


def test_identity_check_accepts_the_expected_digest(corpus):
    path, sha256 = corpus
    assert runner.verify_corpus_identity(path, sha256) == sha256


def test_identity_check_rejects_a_different_corpus(corpus):
    path, _ = corpus
    with pytest.raises(CorpusIdentityError) as excinfo:
        runner.verify_corpus_identity(path, "0" * 64)
    assert "identity mismatch" in str(excinfo.value)


def test_identity_check_reports_a_missing_corpus(tmp_path):
    with pytest.raises(CorpusIdentityError):
        runner.verify_corpus_identity(tmp_path / "absent.smi", "0" * 64)


def test_study_aborts_before_writing_anything_when_the_digest_is_wrong(corpus, tmp_path):
    path, _ = corpus
    output = tmp_path / "study"
    with pytest.raises(CorpusIdentityError):
        run_study(path, output, expected_sha256="0" * 64, progress_every=0)
    assert not output.exists()


def test_study_aborts_when_the_document_count_is_wrong(corpus, tmp_path):
    with pytest.raises(CorpusIdentityError):
        study(corpus, tmp_path, expected_document_count=len(FIXTURE_SMILES) + 1)


def test_the_frozen_corpus_identity_is_pinned():
    """These constants are the Phase 5F-B result the study is defined
    against; drifting them would silently redefine the study."""
    assert (
        runner.FROZEN_FIT_CORPUS_SHA256
        == "b2c4b81160df05c95f8421582bb4b1c95fdf5964a4edaff24a7c1ddd43e2a5de"
    )
    assert runner.FROZEN_DOCUMENT_COUNT == 2_897_639


def test_study_never_modifies_the_corpus(corpus, tmp_path):
    path, sha256 = corpus
    before = path.read_bytes()
    study(corpus, tmp_path)
    assert path.read_bytes() == before
    assert sha256_file(path) == sha256


# ---------------------------------------------------------------------------
# report content and provenance
# ---------------------------------------------------------------------------


def test_report_records_the_verified_digest_and_every_contract(corpus, tmp_path):
    report = study(corpus, tmp_path)
    assert report["corpus"]["verified_sha256"] == corpus[1]
    assert report["corpus"]["normalization_id"] == CANONICAL_SMILES_NORMALIZATION_ID
    assert report["corpus"]["tokenizer_id"] == SMILES_TOKENIZER_ID
    assert report["corpus"]["serialization_id"] == CORPUS_SERIALIZATION_ID
    assert report["corpus"]["uses_downstream_labels"] is False
    assert report["produces_production_artifact"] is False
    assert report["schema_version"] == runner.STUDY_SCHEMA_VERSION
    assert report["run"]["software"]["python"]
    assert report["run"]["software"]["rdkit"]


def test_report_records_the_split_definition_and_its_counts(corpus, tmp_path):
    report = study(corpus, tmp_path)
    split = report["split"]
    assert split["definition"]["bucket_count"] == 20
    assert split["fit_documents"] + split["holdout_documents"] == len(FIXTURE_SMILES)
    assert split["holdout_documents"] > 0
    assert 0.0 < split["holdout_fraction"] < 0.2


def test_report_lists_every_threshold_it_swept(corpus, tmp_path):
    report = study(corpus, tmp_path)
    thresholds = report["definitions"]["thresholds"]
    assert thresholds["min_df"] == [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    assert thresholds["candidate_dimensions"] == [1024, 2048, 4096, 8192, 16384]
    assert thresholds["rarity_df_le"] == [1, 2, 5, 10, 25, 50]


def test_order_tables_account_for_every_occurrence(corpus, tmp_path):
    report = study(corpus, tmp_path)
    for order in ("1", "2", "3"):
        table = report["orders"][order]
        assert (
            table["total_occurrences_fit"] + table["total_occurrences_holdout"]
            == table["total_occurrences_corpus"]
        )
        assert table["distinct_ngrams_corpus"] >= table["distinct_ngrams_fit"]
        assert table["rarity_corpus"]["vocabulary"] == table["distinct_ngrams_corpus"]


def test_min_df_table_combines_orders_additively(corpus, tmp_path):
    report = study(corpus, tmp_path)
    for row in report["min_df_thresholds"]["corpus"]:
        assert row["combined_1_1"] == row["unigrams"]
        assert row["combined_1_2"] == row["unigrams"] + row["bigrams"]
        assert row["combined_1_3"] == row["unigrams"] + row["bigrams"] + row["trigrams"]
        assert row["combined_2_3"] == row["bigrams"] + row["trigrams"]


def test_min_df_vocabulary_sizes_are_non_increasing(corpus, tmp_path):
    report = study(corpus, tmp_path)
    sizes = [row["combined_1_3"] for row in report["min_df_thresholds"]["corpus"]]
    assert sizes == sorted(sizes, reverse=True)


def test_min_df_table_matches_a_hand_counted_accumulator():
    accumulator = NgramFrequencyAccumulator((1, 2, 3))
    accumulator.add_document(("C", "C", "O"), holdout=False)
    accumulator.add_document(("C", "C", "N"), holdout=False)
    accumulator.add_document(("O",), holdout=False)

    entries = {order: accumulator.entries(order) for order in (1, 2, 3)}
    rows = {row["min_df"]: row for row in runner.min_df_table(entries, "corpus")}

    # unigrams: C(df 2), O(df 2), N(df 1)  -> 3 at min_df 1, 2 at min_df 2
    assert rows[1]["unigrams"] == 3
    assert rows[2]["unigrams"] == 2
    # bigrams: (C,C) df 2, (C,O) df 1, (C,N) df 1
    assert rows[1]["bigrams"] == 3
    assert rows[2]["bigrams"] == 1
    # trigrams: (C,C,O) df 1, (C,C,N) df 1
    assert rows[1]["trigrams"] == 2
    assert rows[2]["trigrams"] == 0


def test_every_policy_is_measured(corpus, tmp_path):
    report = study(corpus, tmp_path)
    assert sorted(report["policies"]) == ["A", "B", "C", "D"]
    labels = {policy["label"] for policy in report["policies"].values()}
    assert labels == {"(1,1)", "(1,2)", "(1,3)", "(2,3)"}


def test_holdout_coverage_rows_are_complete_and_bounded(corpus, tmp_path):
    report = study(corpus, tmp_path)
    assert report["holdout_coverage"]
    for row in report["holdout_coverage"]:
        assert 0.0 <= row["holdout_occurrence_coverage"] <= 1.0
        assert 0.0 <= row["holdout_unique_coverage"] <= 1.0
        assert 0.0 <= row["molecule_oov_fraction"]["mean"] <= 1.0
        assert 0 <= row["all_zero_molecules"] <= row["holdout_documents"]
        assert row["nonzero_features"]["max"] <= row["dimension"]
        assert row["ranking"] in ("document_frequency", "term_frequency")


def test_bigram_trigram_policy_reports_no_unigram_verdict(corpus, tmp_path):
    """(2,3) has no unigrams to protect, so "all unigrams retained" is not
    a false answer, it is not a question."""
    report = study(corpus, tmp_path)
    for row in report["holdout_coverage"]:
        if row["policy"] == "(2,3)":
            assert row["all_unigrams_retained"] is None


def test_ranking_comparison_covers_df_versus_tf(corpus, tmp_path):
    report = study(corpus, tmp_path)
    assert report["ranking_comparison"]
    for row in report["ranking_comparison"]:
        assert 0.0 <= row["overlap_fraction"] <= 1.0
        assert row["shared_terms"] <= row["dimension"]


def test_long_molecule_section_quantifies_rank_churn(corpus, tmp_path):
    report = study(corpus, tmp_path)
    section = report["long_molecule_sensitivity"]
    assert section["long_molecule_definition"] == "more than 256 tokens"
    assert len(section["fit_documents_by_band"]) == 6
    assert sum(section["fit_documents_by_band"]) == report["split"]["fit_documents"]
    for row in section["rank_churn_when_long_molecules_dropped"]:
        assert 0.0 <= row["tf_churn_fraction"] <= 1.0
        assert 0.0 <= row["df_churn_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def test_all_study_outputs_are_written(corpus, tmp_path):
    study(corpus, tmp_path)
    written = {path.name for path in (tmp_path / "study").iterdir()}
    assert written == {
        "study_report.json",
        "df_thresholds.csv",
        "vocabulary_coverage.csv",
        "holdout_coverage.csv",
        "top_ngrams.csv",
        "ranking_comparison.csv",
    }


def test_report_on_disk_matches_the_returned_report(corpus, tmp_path):
    report = study(corpus, tmp_path)
    written = (tmp_path / "study" / "study_report.json").read_bytes()
    assert written == study_report_bytes(report)
    assert written.endswith(b"\n")
    assert b"\r\n" not in written


def test_top_ngrams_are_serialized_as_lossless_json_arrays(corpus, tmp_path):
    study(corpus, tmp_path)
    with (tmp_path / "study" / "top_ngrams.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    for row in rows:
        tokens = json.loads(row["ngram"])
        assert isinstance(tokens, list)
        assert len(tokens) == int(row["order"])
        assert all(isinstance(token, str) for token in tokens)


def test_top_ngram_dump_is_bounded(corpus, tmp_path):
    study(corpus, tmp_path)
    with (tmp_path / "study" / "top_ngrams.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) <= 3 * runner.TOP_NGRAM_DIAGNOSTIC_LIMIT


def test_csv_outputs_use_lf_only(corpus, tmp_path):
    study(corpus, tmp_path)
    for name in ("df_thresholds.csv", "holdout_coverage.csv", "ranking_comparison.csv"):
        assert b"\r\n" not in (tmp_path / "study" / name).read_bytes()


def test_existing_output_is_not_overwritten_without_force(corpus, tmp_path):
    study(corpus, tmp_path)
    with pytest.raises(CorpusOutputExistsError):
        study(corpus, tmp_path)
    study(corpus, tmp_path, force=True)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_two_runs_produce_the_same_scientific_result(corpus, tmp_path):
    first = study(corpus, tmp_path, out="one")
    second = study(corpus, tmp_path, out="two")
    assert deterministic_study_view(first) == deterministic_study_view(second)


def test_two_runs_produce_byte_identical_tables(corpus, tmp_path):
    study(corpus, tmp_path, out="one")
    study(corpus, tmp_path, out="two")
    for name in (
        "df_thresholds.csv",
        "vocabulary_coverage.csv",
        "holdout_coverage.csv",
        "top_ngrams.csv",
        "ranking_comparison.csv",
    ):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_only_timings_and_memory_are_excused_from_determinism():
    assert set(runner.STUDY_ID) and runner.STUDY_SCHEMA_VERSION == 1
    from molfusion_backend.corpus.study.report import VOLATILE_STUDY_PATHS

    assert {key for _, key in VOLATILE_STUDY_PATHS} == {
        "started_at",
        "elapsed_seconds",
        "peak_memory_bytes",
        "count_pass_seconds",
        "holdout_pass_seconds",
    }
    assert {section for section, _ in VOLATILE_STUDY_PATHS} == {"run"}


# ---------------------------------------------------------------------------
# unique (distinct) holdout coverage
# ---------------------------------------------------------------------------


def _entry(ngram, fit_df, holdout_df):
    from molfusion_backend.corpus.study.ngrams import NgramEntry

    return NgramEntry(
        ngram=ngram,
        order=len(ngram),
        document_frequency=fit_df + holdout_df,
        term_frequency=fit_df + holdout_df,
        document_frequency_fit=fit_df,
        term_frequency_fit=fit_df,
        document_frequency_holdout=holdout_df,
        term_frequency_holdout=holdout_df,
        document_frequency_fit_bands=(fit_df, 0, 0, 0, 0, 0),
        term_frequency_fit_bands=(fit_df, 0, 0, 0, 0, 0),
    )


def test_unique_coverage_counts_distinct_holdout_ngrams_at_every_prefix():
    """A corpus-level ratio, not an average of per-molecule ones: a motif
    the holdout uses in a thousand molecules still counts once."""
    from molfusion_backend.corpus.study.coverage import VocabularyFamily

    entries = {
        1: [
            _entry(("A",), fit_df=5, holdout_df=1),
            _entry(("B",), fit_df=3, holdout_df=0),
            _entry(("C",), fit_df=1, holdout_df=1),
        ]
    }
    family = VocabularyFamily(
        name="A-df",
        policy="(1,1)",
        orders=(1,),
        ranking="document_frequency",
        protected_unigrams=False,
        sizes=(1, 2, 3),
        ranked=(("A",), ("B",), ("C",)),
    )

    coverage = runner.unique_ngram_coverage([family], entries)

    # The holdout contains two distinct n-grams: ("A",) and ("C",).
    assert coverage[("A-df", 1)] == {
        "holdout_distinct_ngrams": 2,
        "holdout_distinct_covered": 1,
        "holdout_unique_coverage": 0.5,
    }
    # ("B",) never occurs in the holdout, so widening to it adds nothing.
    assert coverage[("A-df", 2)]["holdout_unique_coverage"] == 0.5
    assert coverage[("A-df", 3)]["holdout_unique_coverage"] == 1.0


def test_unique_coverage_spans_every_order_a_policy_uses():
    from molfusion_backend.corpus.study.coverage import VocabularyFamily

    entries = {
        1: [_entry(("A",), fit_df=5, holdout_df=1)],
        2: [_entry(("A", "A"), fit_df=4, holdout_df=1)],
    }
    family = VocabularyFamily(
        name="B-df",
        policy="(1,2)",
        orders=(1, 2),
        ranking="document_frequency",
        protected_unigrams=False,
        sizes=(1, 2),
        ranked=(("A",), ("A", "A")),
    )

    coverage = runner.unique_ngram_coverage([family], entries)
    assert coverage[("B-df", 1)]["holdout_distinct_ngrams"] == 2
    assert coverage[("B-df", 1)]["holdout_unique_coverage"] == 0.5
    assert coverage[("B-df", 2)]["holdout_unique_coverage"] == 1.0


def test_unique_coverage_is_monotone_and_reaches_the_seen_ceiling():
    """The full vocabulary still cannot cover motifs the fit subset never
    saw -- that residue is the vocabulary-level OOV floor."""
    from molfusion_backend.corpus.study.coverage import VocabularyFamily

    entries = {
        1: [
            _entry(("A",), fit_df=5, holdout_df=1),
            _entry(("Z",), fit_df=0, holdout_df=1),  # holdout-only, unrankable
        ]
    }
    family = VocabularyFamily(
        name="A-df",
        policy="(1,1)",
        orders=(1,),
        ranking="document_frequency",
        protected_unigrams=False,
        sizes=(1,),
        ranked=(("A",),),
    )

    coverage = runner.unique_ngram_coverage([family], entries)
    assert coverage[("A-df", 1)]["holdout_distinct_ngrams"] == 2
    assert coverage[("A-df", 1)]["holdout_unique_coverage"] == 0.5
