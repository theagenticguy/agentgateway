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
//! at every level, so unknown fields round-trip untouched and a future beta needs
//! no code change here.
//!
//! The beta list stays in the `anthropic-beta` header. Bedrock rejects
//! `anthropic_beta: ["advanced-tool-use-2025-11-20"]` in an InvokeModel body with
//! 400 `invalid beta flag`, while accepting the same value as a header (probed
//! live 2026-08-30), and the gateway forwards client headers to Bedrock as-is.

use agent_core::strng;
use axum_core::body::Body;
use base64::Engine;
use bytes::Bytes;

use crate::parse::aws_sse;
use crate::types::messages::typed as messages_typed;
use crate::{AIError, StreamingUsageGuard, types};

/// Bedrock requires this in the body; it identifies the Anthropic API contract
/// rather than a model version.
const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Render an Anthropic-native `InvokeModel` body.
///
/// `model` and `stream` move out of the body: Bedrock takes the model in the URL
/// path and selects streaming by action (`invoke` vs `invoke-with-response-stream`),
/// and rejects both fields in the body.
pub fn translate(req: &types::messages::Request) -> Result<Vec<u8>, AIError> {
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
		serde_json::from_slice(&translate(req).expect("render must succeed")).unwrap()
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
	fn unknown_payload_type_gets_no_sse_event_name() {
		// Guessing a name would misdescribe the event; the payload's own `type` stays
		// authoritative.
		assert_eq!(sse_event_name("message_delta"), "message_delta");
		assert_eq!(sse_event_name("some_future_event"), "");
	}
}
