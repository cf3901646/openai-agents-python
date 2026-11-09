//! Own one checked, callback-free submission through native retirement.
//!
//! Opening and public object validation stay in Session. This owner consumes
//! the existing event arbiter and never runs a second model or history loop.

use std::sync::{Arc, Mutex};

use codex_otel::sdk_trace::FutureExt as _;
use pyo3::prelude::*;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{Engine, NativeThread, RuntimeOwned, async_bridge, engine, ops, py_error};

mod reducer;

struct Inputs {
    thread: RuntimeOwned<engine::ThreadState>,
    thread_owner: Arc<Mutex<crate::NativeThreadOwner>>,
    engine_owner: Arc<Mutex<crate::EngineOwner>>,
    params: Value,
    submission_id: String,
    tracing_context: Option<String>,
}

#[pyclass(frozen, module = "agents._native")]
pub(crate) struct NativeRun {
    inputs: Mutex<Option<Inputs>>,
    cancelled: CancellationToken,
}

impl NativeRun {
    pub(crate) fn new(
        thread: &NativeThread,
        engine: &Engine,
        params_json: &str,
        submission_id: String,
        has_home: bool,
        tracing_context: Option<String>,
    ) -> PyResult<Option<Self>> {
        let state = thread.current()?;
        let runtime = engine.current()?;
        if !Arc::ptr_eq(&state.engine, &runtime) {
            return Err(py_error("Native run Engine and Thread differ"));
        }
        // The shared runtime has no per-conversation directory, and its checked
        // initial-turn token proves this is a fresh, host-free conversation.
        if has_home
            || runtime.owns_runtime_services()
            || state.initial_turn.lock().map_err(py_error)?.is_none()
            || state.initial_messages.lock().map_err(py_error)?.is_some()
            || !state
                .thread
                .thread_extension_data()
                .get::<crate::python_tools::PythonTools>()
                .is_some_and(|tools| tools.can_run_without_python())
        {
            return Ok(None);
        }
        let params: Value = serde_json::from_str(params_json).map_err(py_error)?;
        let Some(fields) = params.as_object() else {
            return Err(py_error("Native run parameters must be an object"));
        };
        if fields.keys().any(|name| {
            !matches!(
                name.as_str(),
                "mode" | "input" | "observation_submission_id" | "trace" | "tracing_context"
            )
        }) || params["mode"] != "start_if_idle"
            || !params["input"].as_array().is_some_and(|input| {
                input.len() == 1 && input[0]["type"] == "text" && input[0]["text"].is_string()
            })
        {
            return Ok(None);
        }
        if params["observation_submission_id"].as_str() != Some(submission_id.as_str()) {
            return Err(py_error("Native run submission identity differs"));
        }
        let tracing_context =
            tracing_context.or_else(|| params["tracing_context"].as_str().map(str::to_owned));
        Ok(Some(Self {
            inputs: Mutex::new(Some(Inputs {
                thread: state,
                thread_owner: Arc::clone(&thread.inner),
                engine_owner: Arc::clone(&engine.inner),
                params,
                submission_id,
                tracing_context,
            })),
            cancelled: CancellationToken::new(),
        }))
    }
}

#[pymethods]
impl NativeRun {
    fn cancel(&self) {
        self.cancelled.cancel();
    }

    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inputs = self
            .inputs
            .lock()
            .map_err(py_error)?
            .take()
            .ok_or_else(|| py_error("Native run already started"))?;
        let context = inputs
            .tracing_context
            .as_deref()
            .map(codex_otel::sdk_trace::context)
            .transpose()
            .map_err(py_error)?
            .unwrap_or_default();
        let cancelled = self.cancelled.clone();
        async_bridge::future_factory_into_py(py, move || {
            execute(inputs, cancelled)
                .with_context(codex_otel::sdk_trace::explicit_submission_context(context))
        })
    }
}

impl Drop for NativeRun {
    fn drop(&mut self) {
        self.cancelled.cancel();
    }
}

#[pyclass(frozen, module = "agents._native")]
pub(crate) struct NativeRunArchive {
    observations: reducer::ObservationArchive,
}

#[pymethods]
impl NativeRunArchive {
    #[getter]
    fn sequence(&self) -> u64 {
        self.observations.sequence
    }

    #[getter]
    fn closed(&self) -> bool {
        self.observations.closed
    }

    fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.observations).map_err(py_error)
    }
}

fn failure(message: impl std::fmt::Display) -> Value {
    json!({"message": message.to_string()})
}

fn valid_retry_date(value: &str) -> anyhow::Result<bool> {
    // Reuse the SDK's accepted HTTP-date grammar, including obsolete forms.
    // Numeric retry delays and ordinary rate windows never enter Python here.
    Python::attach(|py| -> PyResult<bool> {
        let parsed = pyo3::types::PyModule::import(py, "agents.rate_limits")?
            .getattr("_retry_after")?
            .call1((value,))?;
        Ok(!parsed.get_item(1)?.is_none())
    })
    .map_err(Into::into)
}

struct RunOutput {
    payload: String,
    archive: NativeRunArchive,
    failures: Vec<(&'static str, PyErr)>,
}

impl<'py> IntoPyObject<'py> for RunOutput {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let failures = self
            .failures
            .into_iter()
            .map(|(phase, error)| (phase, error.value(py).clone().unbind()))
            .collect::<Vec<_>>();
        Ok((self.payload, self.archive, failures)
            .into_pyobject(py)?
            .into_any())
    }
}

async fn execute(inputs: Inputs, cancelled: CancellationToken) -> PyResult<RunOutput> {
    let Inputs {
        thread,
        thread_owner,
        engine_owner,
        params,
        submission_id,
        tracing_context: _,
    } = inputs;
    let mut reduced =
        reducer::RunReducer::new(thread.id.to_string(), submission_id, None, valid_retry_date);
    let mut error = None;
    let mut failures = Vec::new();
    let mut status = "failed";
    let mut stream_failed = false;
    if cancelled.is_cancelled() {
        status = "cancelled";
    } else {
        // Do not drop an in-flight admission. Its receipt identifies any work
        // accepted while cancellation arrived; retirement below joins it.
        let submitted = async {
            let _operation = thread.engine.operation(false)?;
            ops::submit_replaceable(&thread, params).await
        }
        .await;
        match submitted {
            Ok(receipt) if receipt["status"] == "started" => {
                if let Some(turn_id) = receipt["turn_id"].as_str() {
                    if let Err(cause) = reduced.admitted(turn_id.to_owned()) {
                        failures.push(("reader", py_error(cause)));
                    } else {
                        loop {
                            let event = tokio::select! {
                                biased;
                                _ = cancelled.cancelled() => {
                                    status = "cancelled";
                                    break;
                                }
                                event = thread.next_owned_session_event() => event,
                            };
                            match event {
                                Ok(event) => {
                                    if let Err(cause) = reduced.observe(&event) {
                                        stream_failed = true;
                                        failures.push(("reader", py_error(cause)));
                                        break;
                                    }
                                }
                                Err(cause) => {
                                    stream_failed = true;
                                    failures.push(("reader", py_error(cause)));
                                    break;
                                }
                            }
                            if reduced.model_failure().is_some() {
                                break;
                            }
                            if let Some(terminal) = reduced.terminal() {
                                error = terminal.error.clone();
                                status = if terminal.completed && error.is_none() {
                                    "completed"
                                } else if terminal.interrupted {
                                    "cancelled"
                                } else {
                                    "failed"
                                };
                                break;
                            }
                        }
                    }
                } else {
                    error = Some(failure("Native admission omitted its turn identity"));
                }
            }
            Ok(receipt) => {
                error = Some(json!({
                    "message": "Native run was not admitted",
                    "submission": receipt,
                }));
            }
            Err(cause) => failures.push(("submit", py_error(cause))),
        }
    }

    // Keep the existing owner slots and close order. An Engine retry can finish
    // retirement after a Thread close failed; all failures still reach Python.
    let mut cleanup_errors = Vec::new();
    if let Err(cause) = crate::close_thread_owner(Arc::clone(&thread_owner), None).await {
        cleanup_errors.push(cause.to_string());
        failures.push(("thread_close", cause));
    }
    let engine_closed = match crate::close_engine_owner(engine_owner).await {
        Ok(()) => true,
        Err(cause) => {
            cleanup_errors.push(cause.to_string());
            failures.push(("engine_close", cause));
            false
        }
    };
    let thread_closed = match thread_owner.lock() {
        Ok(owner) => owner.live.is_none(),
        Err(cause) => {
            cleanup_errors.push(cause.to_string());
            failures.push(("engine_close", py_error(cause)));
            false
        }
    };
    let retired = engine_closed && thread_closed;
    if retired {
        reduced.set_final_usage(
            thread
                .thread
                .token_usage_info()
                .await
                .map(serde_json::to_value)
                .transpose()
                .map_err(py_error)?,
        );
    }
    // No live native snapshot may escape in a result or its lazy observations.
    drop(thread);
    drop(thread_owner);
    let execution_runtime_error = failures
        .iter()
        .any(|(phase, _)| matches!(*phase, "submit" | "reader"));
    let archive = reduced.finish(
        status,
        match status {
            "completed" => None,
            "cancelled" => Some("TurnStoppedError"),
            _ if execution_runtime_error => Some("RuntimeError"),
            _ => Some("NativeError"),
        },
        if stream_failed { "failed" } else { "closed" },
    );
    let payload = json!({
        "status": status,
        "cancel_requested": cancelled.is_cancelled(),
        "turn_id": archive.turn_id,
        "output": archive.final_output,
        "error": error,
        "terminal": archive.terminal,
        "model_failure": archive.model_failure,
        "cleanup_error": cleanup_errors.last(),
        "cleanup_errors": cleanup_errors,
        "retired": retired,
        "usage": archive.usage,
        "api_rate_limits": archive.api_rate_limits,
    });
    Ok(RunOutput {
        payload: serde_json::to_string(&payload).map_err(py_error)?,
        archive: NativeRunArchive {
            observations: archive.observations,
        },
        failures,
    })
}
