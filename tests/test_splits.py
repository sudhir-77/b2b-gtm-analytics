from src.splits import assert_frames_disjoint, save_splits, time_splits


def test_time_split_is_disjoint_and_holds_out_the_last_three_months():
    splits = time_splits(24)
    assert splits["test"] == [21, 22, 23]
    assert splits["val"] == [18, 19, 20]
    assert 0 not in splits["train"]
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])


def test_saved_split_matches_the_rule(tmp_path):
    path = tmp_path / "splits.json"
    payload = save_splits(path, 24, seed=42)
    assert payload["forecast"]["test_months"] == [21, 22, 23]
    assert payload["forecast"]["feature_origin_month"] == 20
    assert path.exists()


def test_disjoint_ids_reject_overlap():
    import pandas as pd

    assert_frames_disjoint(pd.Series(["a", "b"]), pd.Series(["c"]))
    try:
        assert_frames_disjoint(pd.Series(["a"]), pd.Series(["a"]))
    except ValueError:
        return
    raise AssertionError("overlap was accepted")
