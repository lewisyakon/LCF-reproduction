import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data as data


def test_c2c_message_filter_omits_system_and_strips_only_user_text():
    row = {"conversations": [{"from": "system", "value": "System instruction"},
                             {"from": "human", "value": "  Question?  "},
                             {"from": "gpt", "value": "  Answer.  "}]}
    assert data.messages_of(row) == [{"role": "user", "content": "Question?"},
                                     {"role": "assistant", "content": "  Answer.  "}]


def test_newer_arrow_metadata_can_be_read_without_modifying_cache(tmp_path):
    import hashlib
    import json
    import pyarrow as pa
    rows = [{"conversations": [{"from": "human", "value": "A question"},
                                {"from": "gpt", "value": "An answer"}]}]
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"huggingface": json.dumps({"info": {
        "features": {"conversations": {"_type": "List", "feature": {
            "from": {"_type": "Value", "dtype": "string"},
            "value": {"_type": "Value", "dtype": "string"}}}}}}).encode()})
    path = tmp_path / "newer-cache.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = data.load_cached_openhermes(path)
    assert loaded[0] == rows[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def test_openhermes_prompt_and_response_reconstruct_full_conversation():
    row = {"conversations": [{"from": "human", "value": "What is 2 + 3?"},
                             {"from": "gpt", "value": "The answer is 5."}]}
    prepared = data.openhermes(row, 0)
    assert prepared["status"] == "ok"
    assert prepared["sender_ids"] == prepared["receiver_ids"][:-1]
    expected = data.tokenizer().apply_chat_template(data.messages_of(row), tokenize=True,
        add_generation_prompt=False, enable_thinking=False)
    assert prepared["receiver_ids"] + prepared["response_ids"] == expected


def test_hotpot_partition_is_balanced_disjoint_and_deterministic():
    titles = [f"Paragraph{i}" for i in range(10)]
    row = {"id": "synthetic-check", "question": "Which two facts are needed?", "answer": "SUPERVISION_SENTINEL",
           "context": {"title": titles, "sentences": [[f"Evidence number {i}."] for i in range(10)]},
           "supporting_facts": {"title": titles[:2], "sent_id": [0, 0]}}
    prepared = data.hotpot(row, 0)
    assert prepared == data.hotpot(row, 0)
    assert prepared["status"] == "ok"
    sender = data.tokenizer().decode(prepared["sender_ids"])
    receiver = data.tokenizer().decode(prepared["receiver_ids"])
    s_titles = {title for title in titles if f"[{title}]" in sender}
    r_titles = {title for title in titles if f"[{title}]" in receiver}
    assert len(s_titles) == len(r_titles) == 5
    assert s_titles.isdisjoint(r_titles)
    assert len(s_titles & set(titles[:2])) == len(r_titles & set(titles[:2])) == 1
    assert "SUPERVISION_SENTINEL" not in sender + receiver
    assert len(prepared["spans"]) == 5
    assert all(0 <= a < b <= len(prepared["sender_ids"]) for a, b in prepared["spans"])


def test_reserved_exact_test_match_is_excluded():
    row = {"conversations": [{"from": "human", "value": "Reserved question?"},
                             {"from": "gpt", "value": "Answer."}]}
    previous = data.TEST_HASHES
    try:
        data.TEST_HASHES = {data.digest("Reserved question?")}
        assert data.openhermes(row, 0)["status"] == "test_exact_overlap"
    finally:
        data.TEST_HASHES = previous
