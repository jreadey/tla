from replay_gui import _resolve_belief_path


def test_explicit_belief_flag_wins_over_saved_path():
    records = [{"type": "initial", "belief_path": "saved.h5"}]
    assert _resolve_belief_path("logs/game.jsonl", "explicit.h5", records) == "explicit.h5"


def test_falls_back_to_saved_path_resolved_against_replay_directory():
    records = [{"type": "initial", "belief_path": "game.h5"}]
    assert _resolve_belief_path("logs/game.jsonl", None, records) == "logs/game.h5"


def test_saved_absolute_path_is_returned_unchanged():
    records = [{"type": "initial", "belief_path": "/abs/path/game.h5"}]
    assert _resolve_belief_path("logs/game.jsonl", None, records) == "/abs/path/game.h5"


def test_no_saved_path_and_no_explicit_flag_means_no_overlay():
    records = [{"type": "initial", "belief_path": None}]
    assert _resolve_belief_path("logs/game.jsonl", None, records) is None


def test_an_older_replay_log_with_no_belief_path_key_at_all_is_fine():
    records = [{"type": "initial"}]  # written before belief_path existed
    assert _resolve_belief_path("logs/game.jsonl", None, records) is None


def test_empty_records_list_is_fine():
    assert _resolve_belief_path("logs/game.jsonl", None, []) is None
