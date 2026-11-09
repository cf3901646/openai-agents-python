use super::*;

fn reducer() -> RunReducer {
    RunReducer::new(
        "session".into(),
        "submission".into(),
        Some("turn".into()),
        |_| Ok(false),
    )
}
fn event(kind: &str, fields: Value) -> Value {
    let mut msg = fields;
    msg["type"] = json!(kind);
    json!({"id":"turn", "msg":msg})
}

#[test]
fn unknown_blank_tail_preserves_answer_but_explicit_blank_final_clears_it() {
    for (phase, expected) in [(Value::Null, "answer"), (json!("final_answer"), "")] {
        let mut reducer = reducer();
        reducer
            .observe(&event(
                "raw_response_item",
                json!({"item":{
            "type":"message","role":"assistant","phase":phase,
            "content":[{"type":"output_text","text":" \n"}]}}),
            ))
            .unwrap();
        reducer
            .observe(&event(
                "turn_complete",
                json!({"turn_id":"turn","last_agent_message":"answer","error":null}),
            ))
            .unwrap();
        assert_eq!(
            reducer.finish("completed", None, "closed").final_output,
            expected
        );
    }
}

#[test]
fn retention_is_a_contiguous_suffix_with_original_global_sequences() {
    let mut reducer = reducer();
    for _ in 0..300 {
        reducer
            .observe(&event(
                "agent_message_content_delta",
                json!({"thread_id":"session","turn_id":"turn","item_id":"item","delta":"x"}),
            ))
            .unwrap();
    }
    let archive = reducer.finish("completed", None, "closed");
    assert_eq!(archive.observations.sequence, 303);
    assert_eq!(archive.observations.rows.len(), 256);
    assert_eq!(archive.observations.oldest_sequence, 48);
    assert_eq!(
        archive.observations.rows.last().unwrap()["type"],
        "session_ended"
    );
}

#[test]
fn character_limit_counts_unicode_characters_and_oversize_advances_a_gap() {
    let mut reducer = reducer();
    reducer.observe(&event("agent_message_content_delta", json!({"thread_id":"session","turn_id":"turn","item_id":"item","delta":"🦀".repeat(20_000)}))).unwrap();
    assert_eq!(reducer.rows.len(), 2); // UTF-8 bytes exceed the bound, characters do not.
    reducer.observe(&event("agent_message_content_delta", json!({"thread_id":"session","turn_id":"turn","item_id":"item","delta":"x".repeat(MAX_CHARACTERS)}))).unwrap();
    let archive = reducer.finish("completed", None, "closed");
    assert_eq!(archive.observations.sequence, 5);
    assert_eq!(archive.observations.oldest_sequence, 4);
    assert_eq!(archive.observations.rows.len(), 2);
}

#[test]
fn preadmission_failure_does_not_invent_a_turn_but_finishes_observations() {
    let reducer = RunReducer::new("session".into(), "submission".into(), None, |_| Ok(false));
    let archive = reducer.finish("cancelled", Some("CancelledError"), "failed");
    assert_eq!(archive.observations.sequence, 1);
    assert_eq!(archive.observations.rows.len(), 1);
    assert_eq!(archive.observations.rows[0]["type"], "session_ended");
    assert!(archive.observations.closed);
    assert_eq!(archive.observations.oldest_sequence, 1);
}

#[test]
fn rate_only_update_is_not_a_usage_snapshot_and_final_usage_does_not_publish() {
    let mut reducer = reducer();
    reducer.observe(&event("token_count", json!({"info":null,"rate_limits":null,
        "api_rate_limits":{"requests":{"remaining":3,"reset_after":"1.5s","unrecognized":"discard"}}}))).unwrap();
    reducer.set_final_usage(Some(json!({"model_context_window":123})));
    let archive = reducer.finish("completed", None, "closed");
    assert_eq!(archive.observations.sequence, 4);
    assert_eq!(
        archive.observations.rows[1]["type"],
        "api_rate_limits_updated"
    );
    assert_eq!(
        archive.observations.rows[1]["snapshot"]["requests"]
            .as_object()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(archive.usage.unwrap()["model_context_window"], 123);
}

#[test]
fn response_facts_preserve_typed_records_and_current_identity() {
    Python::initialize();
    let session = "00000000-0000-0000-0000-000000000001";
    let mut reduced = RunReducer::new(
        session.into(),
        "submission".into(),
        Some("turn".into()),
        |_| Ok(false),
    );
    let record = json!({"id":"fact", "thread_id":session, "turn_id":"turn",
        "request_id":"request", "response_id":"response", "upstream_request_id":"upstream",
        "fact":{"kind":"refusal","item_id":null,"content_index":0,
            "snapshot":true,"text":"Synthetic refusal.","capture_complete":true}});
    let mut message = record.clone();
    message["type"] = json!("response_fact");
    let mut foreign = message.clone();
    foreign["thread_id"] = json!("00000000-0000-0000-0000-000000000002");
    reduced
        .observe(&json!({"id":"turn", "msg":foreign}))
        .unwrap();
    let mut other_turn = message.clone();
    other_turn["turn_id"] = json!("other");
    reduced
        .observe(&json!({"id":"other", "msg":other_turn}))
        .unwrap();
    reduced
        .observe(&json!({"id":"turn", "msg":message}))
        .unwrap();
    let archive = reduced.finish("failed", Some("ModelError"), "closed");
    let facts: Vec<_> = archive
        .observations
        .rows
        .iter()
        .filter(|row| row["type"] == "response_fact")
        .collect();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0]["record"], record);
    assert_eq!(facts[0]["submission_id"], "submission");
    assert_eq!(facts[0]["sequence"], 2);
}

#[test]
fn response_fact_record_characters_participate_in_the_public_suffix_bound() {
    Python::initialize();
    let session = "00000000-0000-0000-0000-000000000001";
    let mut reduced = RunReducer::new(
        session.into(),
        "submission".into(),
        Some("turn".into()),
        |_| Ok(false),
    );
    reduced.publish(
        json!({"type":"output_text_delta", "session_id":session,"delta":"x".repeat(64_000)}),
    );
    reduced
        .observe(
            &json!({"id":"turn", "msg":{"type":"response_fact","id":"fact",
        "thread_id":session,"turn_id":"turn","request_id":null,"response_id":null,
        "fact":{"kind":"refusal","item_id":null,"content_index":null,"snapshot":false,
            "text":"é".repeat(2_000),"capture_complete":true}}}),
        )
        .unwrap();
    let archive = reduced.finish("failed", Some("ModelError"), "closed");
    assert_eq!(archive.observations.oldest_sequence, 3);
    assert_eq!(archive.observations.rows.len(), 3);
    assert_eq!(archive.observations.rows[0]["type"], "response_fact");
}

#[test]
fn response_fact_size_uses_python_float_spelling_and_unicode_characters() {
    Python::initialize();
    let record = json!({"logprob":1e-7,"text":"é"});
    assert_eq!(
        response_fact_size(&record).unwrap(),
        "{\"logprob\":1e-07,\"text\":\"é\"}".chars().count()
    );
}
