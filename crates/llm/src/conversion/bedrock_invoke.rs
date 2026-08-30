//! Bedrock `InvokeModel` carrying an Anthropic-native body.
//!
//! Converse cannot express parts of the Anthropic API. Tool search is the clear
//! case: `ToolSpecification` has no `defer_loading`, so schemas cannot be
//! withheld, and `ToolResultContentBlock` has no member for the `tool_reference`
//! blocks a search returns — Bedrock accepts a tool result carrying an
//! unrecognized member with a 200 and silently drops it. Anything that has to
//! survive therefore cannot be translated; it has to be forwarded.
//!
//! So this module deliberately does almost nothing. The request is edited as a
//! JSON map (the shape `from_anthropic_token_count` already uses) instead of
//! being round-tripped through `types::messages::typed`, because those structs
//! narrow the payload: `typed::CustomTool` drops `defer_loading`, and
//! `typed::ContentBlock`'s `#[serde(other)]` arm re-serializes an unknown block
//! as `{"type":"unknown"}` — corrupting a `tool_reference` rather than merely
//! losing it. The weak `types::messages` structs carry `#[serde(flatten)] rest`
//! at every level, so nested content and tool definitions round-trip untouched
//! and a new block or tool field needs no code change here.
//!
//! The **top level** is the exception, because Bedrock's Anthropic schema closes
//! it: an unrecognized key returns 400 `<key>: Extra inputs are not permitted`.
//! Forwarding a full Claude Code body verbatim therefore trades a client-side
//! parse failure for an upstream validation failure — `context_management` is the
//! first field to hit this, and would not be the last. So top-level keys are
//! filtered against [`DEFAULT_INVOKE_BODY_FIELDS`] and dropped keys are logged by
//! name, rather than blacklisting each offender as it surfaces in production.
//!
//! Betas are the other exception, and they move in the opposite direction: the
//! `anthropic-beta` header is the one Bedrock ignores (probed live 2026-08-30 —
//! even an invented value returns 200), and the `anthropic_beta` *body* field is
//! the one it reads and validates, under its own names. A client that believes it
//! is talking to Anthropic puts the list in the header, so the header is lifted
//! into the body and renamed, matching what Claude Code itself sends when it
//! addresses Bedrock directly.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use agent_core::strng;
use axum_core::body::Body;
use base64::Engine;
use bytes::Bytes;

use crate::parse::aws_sse;
use crate::types::messages::typed as messages_typed;
use tracing::{debug, warn};

use crate::{AIError, StreamingUsageGuard, types};

/// Bedrock requires this in the body; it identifies the Anthropic API contract
/// rather than a model version.
const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Top-level body fields Bedrock's Anthropic InvokeModel schema accepts.
///
/// Bedrock closes the top level: an unrecognized key is rejected outright with
/// `<key>: Extra inputs are not permitted`, so forwarding one can never succeed
/// and dropping it is the only behaviour that can. That is the opposite of the
/// nested case — content blocks and tool definitions pass through permissively,
/// which is why `tool_reference` and `defer_loading` survive — so the filter is
/// deliberately confined to this one level.
///
/// Filtering here rather than enumerating known-bad keys is what keeps each new
/// client-side beta field from becoming its own production 400. Claude Code sends
/// several Bedrock does not model (`context_management`, `mcp_servers`,
/// `container`, `betas`, `tool_search_server`); all are dropped by omission.
///
/// The accepted set is model-dependent (probed 2026-08-30: `service_tier` is in
/// Opus 5's schema but not Sonnet 4.6's), and `temperature`/`top_p`/`top_k` are
/// schema-valid but rejected as deprecated by some models — so those stay, and a
/// value-level rejection is left to Bedrock rather than guessed at here.
const DEFAULT_INVOKE_BODY_FIELDS: &[&str] = &[
	"anthropic_beta",
	"anthropic_version",
	"max_tokens",
	"messages",
	"metadata",
	"output_config",
	"stop_sequences",
	"system",
	"temperature",
	"thinking",
	"tool_choice",
	"tools",
	"top_k",
	"top_p",
];

/// Sentinel expanding to [`DEFAULT_INVOKE_BODY_FIELDS`] inside the env override,
/// matching the convention `AGENTGATEWAY_BEDROCK_ANTHROPIC_BETA_HEADERS` uses.
const DEFAULT_SENTINEL: &str = "default";

/// `AGENTGATEWAY_BEDROCK_INVOKE_BODY_FIELDS` overrides the allowlist, so a field
/// Bedrock starts accepting can be let through without a new build.
static ALLOWED_BODY_FIELDS: LazyLock<HashSet<String>> = LazyLock::new(|| {
	match std::env::var("AGENTGATEWAY_BEDROCK_INVOKE_BODY_FIELDS") {
		Ok(raw) => raw
			.split(',')
			.map(str::trim)
			.filter(|f| !f.is_empty())
			.flat_map(|f| {
				if f == DEFAULT_SENTINEL {
					DEFAULT_INVOKE_BODY_FIELDS
						.iter()
						.map(|s| s.to_string())
						.collect::<Vec<_>>()
				} else {
					vec![f.to_string()]
				}
			})
			.collect(),
		Err(_) => DEFAULT_INVOKE_BODY_FIELDS
			.iter()
			.map(|s| s.to_string())
			.collect(),
	}
});

/// Field names already reported, so a dropped field is named once rather than on
/// every request. Bounded by the allowlist's complement, which is tiny in practice.
static REPORTED_DROPS: LazyLock<Mutex<HashSet<String>>> =
	LazyLock::new(|| Mutex::new(HashSet::new()));

/// Drop top-level fields Bedrock will not accept, naming each one the first time.
///
/// A silent drop and a 400 are both failures, but only one is diagnosable; the
/// log line is what makes "this beta is not reaching Bedrock" distinguishable from
/// the next unrelated incident.
fn retain_supported_fields(body: &mut serde_json::Map<String, serde_json::Value>) {
	let dropped: Vec<String> = body
		.keys()
		.filter(|k| !ALLOWED_BODY_FIELDS.contains(k.as_str()))
		.cloned()
		.collect();
	if dropped.is_empty() {
		return;
	}
	for field in &dropped {
		body.remove(field);
	}
	let fresh: Vec<&String> = {
		let mut reported = REPORTED_DROPS.lock().expect("drop set is never poisoned");
		dropped.iter().filter(|f| reported.insert((*f).clone())).collect()
	};
	if !fresh.is_empty() {
		warn!(
			"bedrock invoke: dropping request fields the Anthropic InvokeModel schema \
			 rejects: {}. Set AGENTGATEWAY_BEDROCK_INVOKE_BODY_FIELDS to override.",
			fresh
				.iter()
				.map(|f| f.as_str())
				.collect::<Vec<_>>()
				.join(", ")
		);
	}
	debug!("bedrock invoke: dropped fields {dropped:?}");
}

/// Betas Bedrock names differently from Anthropic.
///
/// Bedrock validates `anthropic_beta` values against its own vocabulary and
/// answers 400 `invalid beta flag` for a name it does not know, so forwarding the
/// Anthropic spelling loses the feature. Claude Code performs this same rename
/// itself when it talks to Bedrock directly (captured 2026-08-30: a request that
/// carries `advanced-tool-use-2025-11-20` first-party carries
/// `tool-search-tool-2025-10-19` to Bedrock), which is what makes this the
/// mapping to match rather than a guess.
const BETA_RENAMES: &[(&str, &str)] = &[(
	"advanced-tool-use-2025-11-20",
	"tool-search-tool-2025-10-19",
)];

/// Betas that mean nothing without a body field [`retain_supported_fields`] removes.
///
/// Advertising one whose field is gone asks Bedrock to apply a feature the request
/// no longer configures — a behaviour change the client never asked for, and one it
/// cannot see. Scoped deliberately to fields the filter itself drops: a beta whose
/// companion field is merely absent may still apply legitimately, so requiring a
/// field the client simply did not send would be a guess.
const BETA_REQUIRES_FIELD: &[(&str, &str)] = &[(
	"context-management-2025-06-27",
	"context_management",
)];

/// Settle `anthropic_beta` from both the body and the `anthropic-beta` header.
///
/// Bedrock ignores the header entirely and reads only the body field (probed
/// 2026-08-30: an unknown header value is accepted, an unknown body value is 400
/// `invalid beta flag`). Claude Code sends the list as a header when it believes
/// it is talking to Anthropic, so leaving the header alone means Bedrock receives
/// no betas at all — measurably unlike direct Bedrock use, which carries them in
/// the body. The Converse path already lifts the header this way into
/// `additionalModelRequestFields`; this is the same move for this path.
///
/// Values are renamed, then filtered against the Converse path's allowlist so
/// there is one place to update, `AGENTGATEWAY_BEDROCK_ANTHROPIC_BETA_HEADERS`
/// included. The body's own values come first and win, since a client that named
/// a beta in the body addressed Bedrock deliberately. Removing the key entirely
/// when nothing survives keeps an empty array from reading as "betas explicitly
/// disabled".
fn merge_beta_values(
	body: &mut serde_json::Map<String, serde_json::Value>,
	headers: &::http::HeaderMap,
) {
	fn rename(value: &str) -> &str {
		BETA_RENAMES
			.iter()
			.find_map(|(from, to)| (*from == value).then_some(*to))
			.unwrap_or(value)
	}

	let from_body: Vec<String> = match body.get("anthropic_beta") {
		Some(serde_json::Value::Array(values)) => values
			.iter()
			.filter_map(|v| v.as_str().map(|s| rename(s).to_owned()))
			.collect(),
		_ => Vec::new(),
	};
	let from_header: Vec<String> = headers
		.get_all("anthropic-beta")
		.iter()
		.filter_map(|v| v.to_str().ok())
		.flat_map(|raw| raw.split(','))
		.map(str::trim)
		.filter(|value| !value.is_empty())
		.map(|value| rename(value).to_owned())
		.collect();

	let allowed = super::bedrock::helpers::allowed_beta_values();
	let orphaned = |value: &str| -> bool {
		BETA_REQUIRES_FIELD
			.iter()
			.any(|(beta, field)| *beta == value && !body.contains_key(*field))
	};

	let mut kept: Vec<serde_json::Value> = Vec::new();
	let mut seen: HashSet<String> = HashSet::new();
	let mut dropped: Vec<String> = Vec::new();
	for value in from_body.into_iter().chain(from_header) {
		if !allowed.iter().any(|a| *a == value) || orphaned(&value) {
			if !dropped.contains(&value) {
				dropped.push(value);
			}
			continue;
		}
		if seen.insert(value.clone()) {
			kept.push(serde_json::Value::String(value));
		}
	}

	if !dropped.is_empty() {
		debug!("bedrock invoke: dropping unusable anthropic_beta values {dropped:?}");
	}
	if kept.is_empty() {
		body.remove("anthropic_beta");
	} else {
		debug!("bedrock invoke: sending anthropic_beta {kept:?}");
		body.insert("anthropic_beta".to_string(), serde_json::Value::Array(kept));
	}
}

/// Render an Anthropic-native `InvokeModel` body.
///
/// `model` and `stream` move out of the body: Bedrock takes the model in the URL
/// path and selects streaming by action (`invoke` vs `invoke-with-response-stream`),
/// and rejects both fields in the body.
pub fn translate(
	req: &types::messages::Request,
	headers: &::http::HeaderMap,
) -> Result<Vec<u8>, AIError> {
	let mut body = match serde_json::to_value(req).map_err(AIError::RequestMarshal)? {
		serde_json::Value::Object(map) => map,
		other => {
			return Err(AIError::RequestMarshal(serde::ser::Error::custom(format!(
				"expected an object body, got {other}"
			))));
		},
	};
	body.remove("model");
	body.remove("stream");
	retain_supported_fields(&mut body);
	merge_beta_values(&mut body, headers);
	body
		.entry("anthropic_version")
		.or_insert_with(|| serde_json::Value::String(BEDROCK_ANTHROPIC_VERSION.to_string()));
	serde_json::to_vec(&body).map_err(AIError::RequestMarshal)
}

/// A non-streaming `InvokeModel` response is already an Anthropic response, so
/// there is nothing to translate; parsing it through the weak `Response` keeps
/// unknown blocks intact via its flattened `rest`.
pub fn translate_response(bytes: &Bytes) -> Result<Box<dyn crate::types::ResponseType>, AIError> {
	let resp: types::messages::Response =
		serde_json::from_slice(bytes).map_err(crate::logged_response_parsing(bytes))?;
	Ok(Box::new(resp))
}

/// Bedrock wraps errors in its own envelope on this path, so reuse the Anthropic
/// error shaping, which passes an already-Anthropic error through untouched.
pub fn translate_error(bytes: &Bytes, status: ::http::StatusCode) -> Result<Bytes, AIError> {
	super::messages::translate_anthropic_error(bytes, status)
}

/// One decoded frame of an `invoke-with-response-stream` body.
pub struct InvokeStreamEvent {
	/// SSE event name, taken from the payload's own `type`.
	pub event: String,
	/// The Anthropic event JSON, forwarded verbatim.
	pub payload: Bytes,
}

impl InvokeStreamEvent {
	/// Decode one AWS event-stream frame.
	///
	/// Unlike converse-stream, every frame here carries `:event-type: chunk` and
	/// the real discriminator lives inside the payload, which Bedrock wraps as
	/// `{"bytes": "<base64 of the Anthropic event>", "p": "<padding>"}`. The
	/// padding key is junk and must be tolerated.
	pub fn deserialize(m: aws_sse::Message) -> anyhow::Result<Self> {
		let get_header = |name: &str| -> Option<String> {
			m.headers()
				.iter()
				.find(|h| h.name().as_str() == name)
				.and_then(|h| h.value().as_string().ok())
				.map(|s| s.as_str().to_owned())
		};
		// Same exception convention as the Converse stream.
		if get_header(":message-type").as_deref() == Some("exception") {
			let exception_type = get_header(":exception-type").unwrap_or_else(|| "unknown".to_owned());
			let error_message = String::from_utf8_lossy(m.payload()).to_string();
			anyhow::bail!("{exception_type}: {error_message}");
		}

		#[derive(serde::Deserialize)]
		struct Envelope {
			bytes: String,
		}
		let envelope: Envelope = serde_json::from_slice(m.payload())?;
		let inner = base64::prelude::BASE64_STANDARD.decode(envelope.bytes.as_bytes())?;

		#[derive(serde::Deserialize)]
		struct Discriminator {
			#[serde(rename = "type")]
			ty: String,
		}
		let event: Discriminator = serde_json::from_slice(&inner)?;
		Ok(Self {
			event: event.ty,
			payload: Bytes::from(inner),
		})
	}
}

/// Re-frame an `invoke-with-response-stream` body as Anthropic SSE.
///
/// The client asked for Anthropic and Bedrock produced Anthropic; only the
/// framing differs (AWS event-stream in, SSE out). Each event's bytes are
/// forwarded verbatim so an event type this build does not know still reaches the
/// client — the usage parse below is observation-only and never gates output.
pub fn translate_stream(
	b: Body,
	buffer_limit: usize,
	mut log: StreamingUsageGuard,
) -> Body {
	aws_sse::transform_multi(b, buffer_limit, move |frame| {
		let event = match InvokeStreamEvent::deserialize(frame) {
			Ok(event) => event,
			Err(e) => {
				tracing::warn!("failed to parse bedrock invoke stream frame: {e}");
				return Vec::new();
			},
		};
		amend_usage(&mut log, &event);
		// `transform_multi` encodes (name, payload) as an SSE event; serializing a
		// RawValue re-emits the original bytes.
		match serde_json::from_slice::<Box<serde_json::value::RawValue>>(&event.payload) {
			Ok(raw) => vec![(sse_event_name(&event.event), raw)],
			Err(e) => {
				tracing::warn!("bedrock invoke stream payload is not valid json: {e}");
				Vec::new()
			},
		}
	})
}

/// The SSE event name for a payload type.
///
/// The name has to be `&'static str` for the encoder while the type arrives from
/// the wire, so the known Anthropic events map to literals. An unrecognized type
/// yields `""`, which the encoder renders as a `data:`-only event: guessing a name
/// would tell the client this is some other event, whereas omitting it leaves the
/// authoritative `type` field in the forwarded payload as the only claim made.
fn sse_event_name(payload_type: &str) -> &'static str {
	match payload_type {
		"message_start" => "message_start",
		"message_delta" => "message_delta",
		"message_stop" => "message_stop",
		"content_block_start" => "content_block_start",
		"content_block_delta" => "content_block_delta",
		"content_block_stop" => "content_block_stop",
		"ping" => "ping",
		"error" => "error",
		_ => "",
	}
}

/// Record token usage for the access log and rate limiting.
///
/// Anthropic reports `input_tokens` exclusive of cache tokens, which is what
/// `CacheTokenConvention::InputExcludesCache` already assumes for Bedrock, so the
/// counts are stored raw and normalized downstream.
fn amend_usage(log: &mut StreamingUsageGuard, event: &InvokeStreamEvent) {
	let Ok(parsed) = serde_json::from_slice::<messages_typed::MessagesStreamEvent>(&event.payload)
	else {
		// An unknown or new event type costs accounting only; the bytes still ship.
		return;
	};
	match parsed {
		messages_typed::MessagesStreamEvent::MessageStart { message } => {
			log.update(|r| {
				r.response.provider_model = Some(strng::new(&message.model));
				r.response.input_tokens = Some(message.usage.input_tokens as u64);
				r.response.output_tokens = Some(message.usage.output_tokens as u64);
				r.response.cached_input_tokens = message.usage.cache_read_input_tokens.map(|v| v as u64);
				r.response.cache_creation_input_tokens =
					message.usage.cache_creation_input_tokens.map(|v| v as u64);
			});
		},
		messages_typed::MessagesStreamEvent::MessageDelta { delta: _, usage } => {
			log.update(|r| {
				// Final counts; message_start's output_tokens is only the first token.
				if let Some(output) = usage.output_tokens {
					r.response.output_tokens = Some(output as u64);
				}
				if let Some(input) = usage.input_tokens {
					r.response.input_tokens = Some(input as u64);
				}
				if let Some(cached) = usage.cache_read_input_tokens {
					r.response.cached_input_tokens = Some(cached as u64);
				}
				if let Some(created) = usage.cache_creation_input_tokens {
					r.response.cache_creation_input_tokens = Some(created as u64);
				}
			});
		},
		_ => {},
	}
}

#[cfg(test)]
mod tests {
	use bytes::BytesMut;
	use serde_json::json;
	use tokio_util::codec::Decoder;

	use super::*;
	use crate::parse::aws_sse::EventStreamCodec;

	/// The tool_result block that provoked this path, verbatim from a captured
	/// Claude Code session. Two tool_reference parts, no other content.
	fn captured_tool_reference_request() -> types::messages::Request {
		serde_json::from_value(json!({
			"model": "claude-fable-5",
			"max_tokens": 32000,
			"stream": true,
			"messages": [{
				"role": "user",
				"content": [{
					"type": "tool_result",
					"tool_use_id": "tooluse_36nMoA0hip1UR8bztp3jzT",
					"content": [
						{"type": "tool_reference", "tool_name": "mcp__example__list_schedules"},
						{"type": "tool_reference", "tool_name": "mcp__example__list_schedule_occurrences"}
					]
				}]
			}]
		}))
		.expect("the captured request must deserialize")
	}

	fn rendered(req: &types::messages::Request) -> serde_json::Value {
		rendered_with(req, &::http::HeaderMap::new())
	}

	fn rendered_with(
		req: &types::messages::Request,
		headers: &::http::HeaderMap,
	) -> serde_json::Value {
		serde_json::from_slice(&translate(req, headers).expect("render must succeed")).unwrap()
	}

	fn beta_headers(values: &[&str]) -> ::http::HeaderMap {
		let mut headers = ::http::HeaderMap::new();
		for value in values {
			headers.append("anthropic-beta", value.parse().unwrap());
		}
		headers
	}

	#[test]
	fn tool_reference_blocks_survive_verbatim() {
		let body = rendered(&captured_tool_reference_request());
		let content = &body["messages"][0]["content"][0]["content"];
		assert_eq!(
			content,
			&json!([
				{"type": "tool_reference", "tool_name": "mcp__example__list_schedules"},
				{"type": "tool_reference", "tool_name": "mcp__example__list_schedule_occurrences"}
			]),
			"tool_reference parts must reach Bedrock unchanged, got {body}"
		);
	}

	#[test]
	fn defer_loading_survives_on_tools() {
		// Tool search is pointless if the deferral flag is dropped: every schema
		// would be sent anyway.
		let req: types::messages::Request = serde_json::from_value(json!({
			"model": "claude-fable-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "hi"}],
			"tools": [
				{"type": "tool_search_tool_bm25", "name": "tool_search_tool_bm25"},
				{"name": "mcp__example__list_schedules", "input_schema": {"type": "object"},
				 "defer_loading": true}
			]
		}))
		.unwrap();
		let body = rendered(&req);
		assert_eq!(body["tools"][1]["defer_loading"], json!(true), "got {body}");
		assert_eq!(
			body["tools"][0]["type"],
			json!("tool_search_tool_bm25"),
			"server tool type must survive, got {body}"
		);
	}

	#[test]
	fn model_and_stream_move_out_of_the_body() {
		// Bedrock takes the model from the URL and picks streaming by action; both
		// fields in the body are rejected.
		let body = rendered(&captured_tool_reference_request());
		assert!(body.get("model").is_none(), "got {body}");
		assert!(body.get("stream").is_none(), "got {body}");
		assert_eq!(body["anthropic_version"], json!("bedrock-2023-05-31"));
	}

	#[test]
	fn beta_list_is_not_injected_into_the_body() {
		// Bedrock answers 400 "invalid beta flag" for advanced-tool-use in an
		// InvokeModel body; the beta belongs in the anthropic-beta header, which the
		// gateway forwards untouched. The Converse path does inject it, so this guards
		// against copying that behaviour here.
		let body = rendered(&captured_tool_reference_request());
		assert!(body.get("anthropic_beta").is_none(), "got {body}");
	}

	#[test]
	fn client_supplied_anthropic_version_is_preserved() {
		let mut req = captured_tool_reference_request();
		req.rest = json!({"anthropic_version": "bedrock-2023-05-31-custom"});
		let body = rendered(&req);
		assert_eq!(body["anthropic_version"], json!("bedrock-2023-05-31-custom"));
	}

	/// Decode every frame of a real `invoke-with-response-stream` capture.
	fn decode_fixture() -> Vec<InvokeStreamEvent> {
		let raw = std::fs::read("src/tests/response/bedrock/invoke-tool-reference.bin")
			.expect("fixture must exist");
		let mut buf = BytesMut::from(&raw[..]);
		let mut codec = EventStreamCodec::new();
		let mut out = Vec::new();
		while let Some(frame) = codec.decode(&mut buf).expect("framing must decode") {
			out.push(InvokeStreamEvent::deserialize(frame).expect("frame must parse"));
		}
		out
	}

	#[test]
	fn stream_frames_decode_to_anthropic_events() {
		let events = decode_fixture();
		let names: Vec<&str> = events.iter().map(|e| e.event.as_str()).collect();
		assert_eq!(
			names,
			vec![
				"message_start",
				"content_block_start",
				"content_block_delta",
				"content_block_delta",
				"content_block_delta",
				"content_block_delta",
				"content_block_delta",
				"content_block_stop",
				"content_block_start",
				"content_block_delta",
				"content_block_stop",
				"message_delta",
				"message_stop",
			]
		);
		// Every frame is `:event-type: chunk`, so the name must come from the payload,
		// not the frame header.
		for event in &events {
			let payload: serde_json::Value = serde_json::from_slice(&event.payload).unwrap();
			assert_eq!(payload["type"], json!(event.event));
		}
	}

	#[test]
	fn stream_carries_the_tool_made_available_by_the_reference() {
		// The captured turn only reaches this tool through a tool_reference: it was
		// declared defer_loading, so its schema was withheld.
		let events = decode_fixture();
		let tool_names: Vec<String> = events
			.iter()
			.filter_map(|e| {
				let v: serde_json::Value = serde_json::from_slice(&e.payload).ok()?;
				if v["content_block"]["type"] != "tool_use" {
					return None;
				}
				Some(v["content_block"]["name"].as_str()?.to_string())
			})
			.collect();
		assert_eq!(
			tool_names,
			vec!["mcp__example__zqx7_list_widget_manifests_v9".to_string()]
		);
	}

	#[test]
	fn stream_usage_is_recorded_with_cache_fields() {
		let events = decode_fixture();
		let start: serde_json::Value =
			serde_json::from_slice(&events[0].payload).expect("message_start");
		let usage = &start["message"]["usage"];
		// Anthropic reports input exclusive of cache, matching
		// CacheTokenConvention::InputExcludesCache for Bedrock.
		assert!(usage["input_tokens"].as_u64().unwrap() > 0);
		assert!(usage.get("cache_read_input_tokens").is_some());
		let delta: serde_json::Value = serde_json::from_slice(&events[11].payload).expect("delta");
		assert!(delta["usage"]["output_tokens"].as_u64().unwrap() > 0);
	}

	#[test]
	fn context_management_is_dropped() {
		// Reported blocker: a full Claude Code request carries context_management, and
		// Bedrock rejects it with "context_management: Extra inputs are not permitted"
		// -- unconditionally, including under context-management-2025-06-27 and as an
		// empty object (probed 2026-08-30). Forwarding it can only ever 400.
		let mut req = captured_tool_reference_request();
		req.rest = json!({
			"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
		});
		let body = rendered(&req);
		assert!(body.get("context_management").is_none(), "got {body}");
		// The rest of the request must be untouched by the filtering.
		assert_eq!(
			body["messages"][0]["content"][0]["content"][0]["type"],
			json!("tool_reference"),
			"got {body}"
		);
	}

	#[test]
	fn every_field_bedrock_closes_the_door_on_is_dropped() {
		// Bedrock closes the top level: each of these returns "<key>: Extra inputs are
		// not permitted", so they are dropped by omission from the allowlist rather
		// than blacklisted one incident at a time.
		let mut req = captured_tool_reference_request();
		req.rest = json!({
			"context_management": {"edits": []},
			"mcp_servers": [],
			"container": "c-1",
			"betas": ["advanced-tool-use-2025-11-20"],
			"tool_search_server": {},
			"some_field_invented_after_this_build": true
		});
		let body = rendered(&req);
		for key in [
			"context_management",
			"mcp_servers",
			"container",
			"betas",
			"tool_search_server",
			"some_field_invented_after_this_build",
		] {
			assert!(body.get(key).is_none(), "{key} should be dropped, got {body}");
		}
	}

	#[test]
	fn schema_valid_fields_are_kept() {
		// The filter must not overreach. temperature/top_p/top_k are in Bedrock's
		// schema even though some models reject them as deprecated -- that is a
		// value-level answer for Bedrock to give, not something to guess at here.
		let mut req = captured_tool_reference_request();
		req.rest = json!({
			"temperature": 0.5,
			"top_p": 0.9,
			"top_k": 40,
			"stop_sequences": ["STOP"],
			"metadata": {"user_id": "u1"},
			"thinking": {"type": "disabled"},
			"output_config": {"effort": "high"},
			"tool_choice": {"type": "auto"},
			"tools": []
		});
		let body = rendered(&req);
		for key in [
			"temperature",
			"top_p",
			"top_k",
			"stop_sequences",
			"metadata",
			"thinking",
			"output_config",
			"tool_choice",
			"tools",
		] {
			assert!(body.get(key).is_some(), "{key} should survive, got {body}");
		}
		assert!(body.get("system").is_none(), "absent fields stay absent: {body}");
	}

	#[test]
	fn unrecognized_beta_values_are_filtered_from_the_body() {
		// Bedrock validates body beta values and uses its own names: tool search is
		// tool-search-tool-2025-10-19 here, not advanced-tool-use-2025-11-20. An
		// unfiltered client list would 400 the whole request.
		let mut req = captured_tool_reference_request();
		req.rest = json!({
			"anthropic_beta": ["advanced-tool-use-2025-11-20", "tool-search-tool-2025-10-19"]
		});
		let body = rendered(&req);
		assert_eq!(
			body["anthropic_beta"],
			json!(["tool-search-tool-2025-10-19"]),
			"got {body}"
		);
	}

	#[test]
	fn beta_key_is_removed_when_no_value_survives() {
		// An empty array would read as "betas explicitly disabled" rather than "the
		// client asked for betas Bedrock does not know".
		let mut req = captured_tool_reference_request();
		req.rest = json!({"anthropic_beta": ["zz-invented-9999-01-01"]});
		let body = rendered(&req);
		assert!(body.get("anthropic_beta").is_none(), "got {body}");
	}

	#[test]
	fn header_betas_are_lifted_into_the_body() {
		// Bedrock reads only the body field, so a client that sends the list as a
		// header reaches Bedrock with no betas unless the header is lifted.
		let body = rendered_with(
			&captured_tool_reference_request(),
			&beta_headers(&["context-1m-2025-08-07,interleaved-thinking-2025-05-14"]),
		);
		assert_eq!(
			body["anthropic_beta"],
			json!(["context-1m-2025-08-07", "interleaved-thinking-2025-05-14"]),
			"got {body}"
		);
	}

	#[test]
	fn header_betas_are_renamed_to_bedrock_names() {
		// Bedrock answers 400 `invalid beta flag` for Anthropic's spelling of tool
		// search, so forwarding it unmapped loses the feature.
		let body = rendered_with(
			&captured_tool_reference_request(),
			&beta_headers(&["advanced-tool-use-2025-11-20"]),
		);
		assert_eq!(
			body["anthropic_beta"],
			json!(["tool-search-tool-2025-10-19"]),
			"got {body}"
		);
	}

	#[test]
	fn header_betas_the_gateway_does_not_know_are_dropped_not_forwarded() {
		// An unrecognized body value takes the whole request down with a 400, so an
		// unknown header value must not become one.
		let body = rendered_with(
			&captured_tool_reference_request(),
			&beta_headers(&["claude-code-20250219", "context-1m-2025-08-07"]),
		);
		assert_eq!(body["anthropic_beta"], json!(["context-1m-2025-08-07"]), "got {body}");
	}

	#[test]
	fn body_betas_win_over_the_header_and_do_not_duplicate() {
		// A client that named a beta in the body addressed Bedrock deliberately, and
		// the same beta arriving twice is still one beta.
		let mut req = captured_tool_reference_request();
		req.rest = json!({"anthropic_beta": ["tool-search-tool-2025-10-19"]});
		let body = rendered_with(
			&req,
			&beta_headers(&["advanced-tool-use-2025-11-20,context-1m-2025-08-07"]),
		);
		assert_eq!(
			body["anthropic_beta"],
			json!(["tool-search-tool-2025-10-19", "context-1m-2025-08-07"]),
			"got {body}"
		);
	}

	#[test]
	fn context_management_beta_goes_when_its_field_is_filtered_out() {
		// Bedrock rejects the `context_management` field, so it is dropped; keeping
		// its beta would ask for server-side context management the request no longer
		// configures.
		let mut req = captured_tool_reference_request();
		req.rest = json!({"context_management": {"edits": [{"type": "clear_thinking_20251015"}]}});
		let body = rendered_with(
			&req,
			&beta_headers(&["context-management-2025-06-27,context-1m-2025-08-07"]),
		);
		assert!(body.get("context_management").is_none(), "got {body}");
		assert_eq!(body["anthropic_beta"], json!(["context-1m-2025-08-07"]), "got {body}");
	}

	#[test]
	fn captured_claude_code_header_set_renders_the_captured_bedrock_body_set() {
		// The exact header list Claude Code sends a gateway, and the exact
		// `anthropic_beta` array it sends Bedrock directly, both captured 2026-08-30
		// from one session. Matching direct Bedrock use is the whole point of the
		// lift, so the mapping is pinned to the observation rather than to intent.
		let body = rendered_with(
			&captured_tool_reference_request(),
			&beta_headers(&[
				"claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,\
				 thinking-token-count-2026-05-13,context-management-2025-06-27,\
				 prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,\
				 advanced-tool-use-2025-11-20,effort-2025-11-24,fallback-credit-2026-06-01,\
				 extended-cache-ttl-2025-04-11",
			]),
		);
		let rendered_betas: Vec<&str> = body["anthropic_beta"]
			.as_array()
			.expect("anthropic_beta must be an array")
			.iter()
			.map(|v| v.as_str().unwrap())
			.collect();
		for expected in [
			"context-1m-2025-08-07",
			"interleaved-thinking-2025-05-14",
			"tool-search-tool-2025-10-19",
			"mid-conversation-system-2026-04-07",
			"fallback-credit-2026-06-01",
		] {
			assert!(
				rendered_betas.contains(&expected),
				"{expected} is in the body Claude Code sends Bedrock directly, got {body}"
			);
		}
	}

	#[test]
	fn unknown_payload_type_gets_no_sse_event_name() {
		// Guessing a name would misdescribe the event; the payload's own `type` stays
		// authoritative.
		assert_eq!(sse_event_name("message_delta"), "message_delta");
		assert_eq!(sse_event_name("some_future_event"), "");
	}
}
