//! Detached facts and the bounded public suffix for the no-tools native run owner.
use std::collections::VecDeque;

use anyhow::{Result, bail, ensure};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde::Serialize;
use serde_json::{Map, Value, json};

const MAX_ROWS: usize = 256;
const MAX_CHARACTERS: usize = 65_536;

#[derive(Debug, Serialize)]
pub struct TerminalFacts {
    pub completed: bool,
    pub interrupted: bool,
    pub error: Option<Value>,
    pub last_agent_message: Option<String>,
}

/// Only detached values. The existing Python EventStream owns cursor policy.
#[derive(Debug, Serialize)]
pub struct ObservationArchive {
    pub session_id: String,
    pub sequence: u64,
    pub oldest_sequence: u64,
    pub closed: bool,
    pub rows: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct RunArchive {
    pub session_id: String,
    pub submission_id: String,
    pub turn_id: Option<String>,
    pub final_output: String,
    pub usage: Option<Value>,
    pub api_rate_limits: Option<Value>,
    pub model_failure: Option<Value>,
    pub terminal: Option<TerminalFacts>,
    pub observations: ObservationArchive,
}

pub struct RunReducer {
    session_id: String,
    submission_id: String,
    turn_id: Option<String>,
    terminal: Option<TerminalFacts>,
    model_failure: Option<Value>,
    usage: Option<Value>,
    api_rate_limits: Option<Value>,
    blank_final: bool,
    sequence: u64,
    characters: usize,
    rows: VecDeque<(Value, usize)>,
    valid_retry_date: fn(&str) -> Result<bool>,
}

impl RunReducer {
    pub fn new(
        session_id: String,
        submission_id: String,
        turn_id: Option<String>,
        valid_retry_date: fn(&str) -> Result<bool>,
    ) -> Self {
        let mut reducer = Self {
            session_id,
            submission_id,
            turn_id: None,
            terminal: None,
            model_failure: None,
            usage: None,
            api_rate_limits: None,
            blank_final: false,
            sequence: 0,
            characters: 0,
            rows: VecDeque::new(),
            valid_retry_date,
        };
        if let Some(id) = turn_id {
            reducer.admitted(id).expect("first admission");
        }
        reducer
    }

    pub fn admitted(&mut self, turn_id: String) -> Result<()> {
        if let Some(current) = &self.turn_id {
            ensure!(current == &turn_id, "native run cannot admit a second turn");
            return Ok(());
        }
        ensure!(!turn_id.is_empty(), "native turn ID must not be empty");
        self.turn_id = Some(turn_id);
        self.publish(self.submission_row("turn_started"));
        Ok(())
    }

    pub fn terminal(&self) -> Option<&TerminalFacts> {
        self.terminal.as_ref()
    }
    pub fn model_failure(&self) -> Option<&Value> {
        self.model_failure.as_ref()
    }
    pub fn set_final_usage(&mut self, usage: Option<Value>) {
        self.usage = usage;
    }

    /// Consume one current-arbiter value; never retain the raw event stream.
    pub fn observe(&mut self, event: &Value) -> Result<()> {
        if !event.get("subagent_id").unwrap_or(&Value::Null).is_null() {
            return Ok(());
        }
        let message = event
            .get("msg")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("native event has no message"))?;
        let kind = message.get("type").and_then(Value::as_str).unwrap_or("");
        let Some(turn_id) = self.turn_id.as_deref() else {
            return Ok(());
        };
        let id = event.get("id").and_then(Value::as_str);
        let routed_turn = message
            .get("turn_id")
            .or_else(|| message.get("turnId"))
            .and_then(Value::as_str);
        if routed_turn.is_some_and(|id| id != turn_id) {
            return Ok(());
        }
        if kind == "response_fact" {
            // Preserve Core's typed record and the same current-thread/turn
            // routing used by Session._observe_progress.
            if message.get("thread_id").and_then(Value::as_str) == Some(self.session_id.as_str())
                && routed_turn == Some(turn_id)
            {
                let record: codex_protocol::response_facts::ResponseFactRecord =
                    serde_json::from_value(Value::Object(message.clone()))?;
                let mut row = self.submission_row("response_fact");
                let record = serde_json::to_value(record)?;
                // The arbiter receive has returned and no Engine gate or
                // observation lock is held. Match the current Python budget,
                // including its float spelling, only for this rare event.
                let record_size = response_fact_size(&record)?;
                row["record"] = record;
                self.publish_with_record_size(row, record_size);
            }
            return Ok(());
        }
        if kind == "_sdk_request_metadata" {
            if id == Some(turn_id)
                && message.get("submission_id").and_then(Value::as_str)
                    == Some(self.submission_id.as_str())
            {
                if let Some(row) =
                    self.request_observation(message.get("observation").unwrap_or(&Value::Null))
                {
                    self.publish(row);
                }
            }
            return Ok(());
        }
        if matches!(kind, "task_complete" | "turn_complete" | "turn_aborted") {
            if routed_turn.or(id) != Some(turn_id) {
                return Ok(());
            }
            let error = message
                .get("error")
                .filter(|v| !v.is_null())
                .cloned()
                .or_else(|| {
                    message
                        .get("_sdk_policy_failure")
                        .filter(|v| !v.is_null())
                        .cloned()
                });
            if self.terminal.is_none() {
                self.terminal = Some(TerminalFacts {
                    completed: kind != "turn_aborted",
                    interrupted: kind == "turn_aborted",
                    error,
                    last_agent_message: message
                        .get("last_agent_message")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                });
            }
            return Ok(());
        }
        if kind == "_sdk_model_generation_failed" && id == Some(turn_id) {
            if self.model_failure.is_none() {
                self.model_failure = Some(Value::Object(message.clone()));
            }
            return Ok(());
        }
        if kind == "raw_response_item" && id == Some(turn_id) {
            let item = message.get("item").unwrap_or(&Value::Null);
            if item["type"] == "message" && item["role"] == "assistant" {
                let texts: Vec<&str> = item["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|part| {
                        matches!(part["type"].as_str(), Some("input_text" | "output_text"))
                    })
                    .filter_map(|part| part["text"].as_str())
                    .collect();
                let blank = !texts.is_empty() && texts.iter().all(|text| text.trim().is_empty());
                if !blank || item["phase"] == "final_answer" {
                    self.blank_final = blank;
                }
            }
            return Ok(());
        }
        if kind == "token_count" && id == Some(turn_id) {
            let snapshot = match message.get("api_rate_limits") {
                Some(value) => rate_snapshot(value, self.valid_retry_date)?,
                None => None,
            };
            if let Some(snapshot) = snapshot {
                self.api_rate_limits = Some(snapshot.clone());
                let mut row = self.submission_row("api_rate_limits_updated");
                row["snapshot"] = snapshot;
                self.publish(row);
            }
            if message.contains_key("api_rate_limits")
                && message.get("info").unwrap_or(&Value::Null).is_null()
                && message.get("rate_limits").unwrap_or(&Value::Null).is_null()
            {
                return Ok(());
            }
            let usage = message.get("info").filter(|v| !v.is_null()).cloned();
            self.usage = usage.clone();
            let mut row = self.submission_row("usage_updated");
            row["usage"] = usage.unwrap_or(Value::Null);
            self.publish(row);
            return Ok(());
        }
        let correct_item = message.get("thread_id").and_then(Value::as_str)
            == Some(self.session_id.as_str())
            && routed_turn == self.turn_id.as_deref();
        if kind == "agent_message_content_delta" && correct_item {
            if let (Some(item), Some(delta)) = (
                message.get("item_id").and_then(Value::as_str),
                message.get("delta").and_then(Value::as_str),
            ) {
                let mut row = self.submission_row("output_text_delta");
                row["item_id"] = json!(item);
                row["delta"] = json!(delta);
                self.publish(row);
            }
        } else if kind == "item_completed" && correct_item {
            let item = message.get("item").unwrap_or(&Value::Null);
            if item["type"] == "ContextCompaction" {
                if let Some(id) = item["id"].as_str() {
                    let mut row = self.submission_row("context_compacted");
                    row["item_id"] = json!(id);
                    self.publish(row);
                }
            } else if let Some(projected) = annotated_message(item, turn_id) {
                let mut row = self.submission_row("model_output_item_completed");
                row["item"] = projected;
                self.publish(row);
            }
        } else if matches!(
            kind,
            "dynamic_tool_call_request"
                | "exec_approval_request"
                | "apply_patch_approval_request"
                | "_sdk_prepare_request"
                | "_sdk_select_step"
        ) {
            bail!("native run received an unsupported interaction");
        }
        Ok(())
    }

    /// Call after the sole reader ends. Observation completion is independent
    /// of retained cleanup authority, including a failed native close.
    pub fn finish(
        mut self,
        status: &str,
        error_type: Option<&str>,
        session_reason: &str,
    ) -> RunArchive {
        if self.turn_id.is_some() {
            let mut row = self.submission_row("turn_finished");
            row["status"] = json!(status);
            row["error_type"] = json!(error_type);
            self.publish(row);
        }
        self.publish(
            json!({"type":"session_ended","session_id":self.session_id,"reason":session_reason}),
        );
        let final_output = if self.blank_final {
            String::new()
        } else {
            self.terminal
                .as_ref()
                .and_then(|terminal| terminal.last_agent_message.clone())
                .unwrap_or_default()
        };
        let oldest_sequence = self
            .rows
            .front()
            .and_then(|(row, _)| row["sequence"].as_u64())
            .unwrap_or(self.sequence + 1);
        let observations = ObservationArchive {
            session_id: self.session_id.clone(),
            sequence: self.sequence,
            oldest_sequence,
            closed: true,
            rows: self.rows.into_iter().map(|(row, _)| row).collect(),
        };
        RunArchive {
            session_id: self.session_id,
            submission_id: self.submission_id,
            turn_id: self.turn_id,
            final_output,
            usage: self.usage,
            api_rate_limits: self.api_rate_limits,
            model_failure: self.model_failure,
            terminal: self.terminal,
            observations,
        }
    }

    fn submission_row(&self, kind: &str) -> Value {
        json!({"type":kind,"session_id":self.session_id,"submission_id":self.submission_id,
            "turn_id":self.turn_id,"parent_submission_id":null})
    }
    fn publish(&mut self, row: Value) {
        self.publish_with_record_size(row, 0);
    }

    fn publish_with_record_size(&mut self, mut row: Value, record_size: usize) {
        self.sequence += 1;
        row["sequence"] = json!(self.sequence);
        let mut size: usize = row
            .as_object()
            .unwrap()
            .values()
            .filter_map(Value::as_str)
            .map(|v| v.chars().count())
            .sum();
        if let Some(item) = row.get("item") {
            size += serde_json::to_string(item)
                .expect("JSON value")
                .chars()
                .count();
        }
        size += record_size;
        if size > MAX_CHARACTERS {
            self.rows.clear();
            self.characters = 0;
            return;
        }
        self.characters += size;
        self.rows.push_back((row, size));
        while self.rows.len() > MAX_ROWS || self.characters > MAX_CHARACTERS {
            self.characters -= self.rows.pop_front().unwrap().1;
        }
    }

    fn request_observation(&self, value: &Value) -> Option<Value> {
        let value = value.as_object()?;
        for key in ["type", "request_id", "thread_id", "turn_id"] {
            if value.get(key)?.as_str()?.is_empty() {
                return None;
            }
        }
        if value.get("thread_id")?.as_str()? != self.session_id
            || value.get("turn_id")?.as_str()? != self.turn_id.as_deref()?
        {
            return None;
        }
        let kind = value.get("type")?.as_str()?;
        let mut row = self.submission_row(kind);
        for key in ["request_id", "thread_id"] {
            row[key] = value.get(key)?.clone();
        }
        if kind == "model_response_linked" {
            if value.len() != 5 || value.get("response_observation_id")?.as_str()?.is_empty() {
                return None;
            }
            row["response_observation_id"] = value.get("response_observation_id")?.clone();
            return Some(row);
        }
        if kind != "model_request_dispatching" || value.len() != 10 {
            return None;
        }
        value.get("model")?.as_str()?;
        if !matches!(value.get("wire")?.as_str()?, "responses" | "responses_lite")
            || !matches!(
                value.get("transport")?.as_str()?,
                "http_sse" | "http_json" | "websocket"
            )
        {
            return None;
        }
        for key in ["reasoning_summary", "service_tier"] {
            let v = value.get(key)?;
            if !v.is_null() && !v.is_string() {
                return None;
            }
        }
        let effort = value.get("reasoning_effort")?;
        row["reasoning_effort"] = if effort.is_null() {
            Value::Null
        } else {
            let effort = effort.as_object()?;
            match effort.get("kind")?.as_str()? {
                "text" if effort.len() == 2 => json!(effort.get("value")?.as_str()?),
                "integer" if effort.len() == 2 => {
                    let digits = effort.get("decimal")?.as_str()?;
                    if digits.is_empty()
                        || digits.len() > 20
                        || !digits.bytes().all(|c| c.is_ascii_digit())
                        || (digits.len() > 1 && digits.starts_with('0'))
                    {
                        return None;
                    }
                    json!(digits.parse::<u64>().ok()?)
                }
                _ => return None,
            }
        };
        for key in [
            "model",
            "reasoning_summary",
            "service_tier",
            "wire",
            "transport",
        ] {
            row[key] = value.get(key)?.clone();
        }
        Some(row)
    }
}

fn annotated_message(item: &Value, turn_id: &str) -> Option<Value> {
    if item["type"] != "AgentMessage" {
        return None;
    }
    let mut annotated = false;
    let content: Vec<Value> = item["content"]
        .as_array()?
        .iter()
        .filter(|part| part["type"] == "Text")
        .map(|part| {
            let mut text = json!({"type":"output_text","text":part["text"]});
            if !part["annotations"].is_null() {
                text["annotations"] = part["annotations"].clone();
                annotated |= match &part["annotations"] {
                    Value::Null => false,
                    Value::Bool(v) => *v,
                    Value::Array(v) => !v.is_empty(),
                    Value::Object(v) => !v.is_empty(),
                    Value::String(v) => !v.is_empty(),
                    Value::Number(v) => v.as_f64() != Some(0.0),
                };
            }
            text
        })
        .collect();
    if !annotated {
        return None;
    }
    let mut projected = json!({"type":"message","id":item["id"],"turn_id":turn_id,"role":"assistant","status":"completed","content":content,"phase":item["phase"]});
    if !item["display_text"].is_null() {
        projected["display_text"] = item["display_text"].clone();
    }
    Some(projected)
}

fn rate_snapshot(
    value: &Value,
    valid_retry_date: fn(&str) -> Result<bool>,
) -> Result<Option<Value>> {
    let Some(value) = value.as_object() else {
        return Ok(None);
    };
    let mut snapshot = Map::new();
    for key in ["requests", "tokens", "project_tokens"] {
        if let Some(window) = value.get(key).and_then(Value::as_object) {
            let mut kept = Map::new();
            for field in ["limit", "remaining"] {
                if let Some(v) = window.get(field).filter(|v| v.as_u64().is_some()) {
                    kept.insert(field.into(), v.clone());
                }
            }
            if let Some(reset) = window
                .get("reset_after")
                .and_then(Value::as_str)
                .filter(|v| duration(v))
            {
                kept.insert("reset_after".into(), json!(reset));
            }
            if !kept.is_empty() {
                snapshot.insert(key.into(), Value::Object(kept));
            }
        }
    }
    if let Some(retry) = value
        .get("retry_after")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty() && v.chars().count() <= 64)
    {
        if retry.bytes().all(|c| c.is_ascii_digit()) || valid_retry_date(retry)? {
            snapshot.insert("retry_after".into(), json!(retry));
        }
    }
    Ok((!snapshot.is_empty()).then_some(Value::Object(snapshot)))
}
fn duration(value: &str) -> bool {
    if value.is_empty() || value.len() > 64 || !value.is_ascii() {
        return false;
    }
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return false;
        }
        if bytes.get(i) == Some(&b'.') {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == start {
                return false;
            }
        }
        if matches!(bytes.get(i), Some(b'm' | b'u' | b'n')) && bytes.get(i + 1) == Some(&b's') {
            i += 2;
        } else if matches!(bytes.get(i), Some(b'h' | b'm' | b's')) {
            i += 1;
        } else {
            return false;
        }
    }
    true
}

// Core already supplied a typed record. Parse only into built-in Python JSON
// values; stdlib dumps/len is the same character rule as _ObservationBuffer.
fn response_fact_size(record: &Value) -> Result<usize> {
    let encoded = serde_json::to_string(record)?;
    Python::attach(|py| -> PyResult<usize> {
        let json = PyModule::import(py, "json")?;
        let value = json.call_method1("loads", (encoded,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("ensure_ascii", false)?;
        kwargs.set_item("separators", (",", ":"))?;
        json.getattr("dumps")?.call((value,), Some(&kwargs))?.len()
    })
    .map_err(Into::into)
}

#[cfg(test)]
#[path = "reducer_tests.rs"]
mod tests;
