use std::sync::Arc;

use anyhow::Result;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::{McpServer, jsonrpc_error, parse_json_line};

/// Run the MCP server over stdio (line-delimited JSON-RPC).
pub async fn run_stdio(server: McpServer) -> Result<()> {
    let server = Arc::new(server);
    let stdin = io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let message = match parse_json_line(line) {
            Ok(message) => message,
            Err(error) => {
                let response = jsonrpc_error(
                    serde_json::Value::Null,
                    -32700,
                    &format!("parse error: {error}"),
                );
                let payload = serde_json::to_string(&response)?;
                stdout.write_all(payload.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        if let Some(response) = server.handle_message(message).await {
            let payload = serde_json::to_string(&response)?;
            stdout.write_all(payload.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

/// Run the MCP server over SSE (Server-Sent Events) HTTP transport.
///
/// Requires the `sse` feature.
#[cfg(not(feature = "sse"))]
pub async fn run_sse(_server: McpServer, _port: u16) -> Result<()> {
    anyhow::bail!("SSE transport is disabled in this build (enable the `llm-mcp/sse` feature)")
}

#[cfg(feature = "sse")]
pub async fn run_sse(server: McpServer, port: u16) -> Result<()> {
    use hyper::service::service_fn;
    use hyper_util::rt::TokioIo;

    let server = Arc::new(server);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let (events, _) = tokio::sync::broadcast::channel::<String>(4096);

    eprintln!("mcp sse listening on http://127.0.0.1:{port}");
    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let server = Arc::clone(&server);
        let events = events.clone();

        tokio::spawn(async move {
            let service = service_fn(move |request| {
                handle_sse_request(request, Arc::clone(&server), events.clone())
            });
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                eprintln!("mcp sse connection error: {error}");
            }
        });
    }
}

#[cfg(feature = "sse")]
async fn handle_sse_request(
    request: hyper::Request<hyper::body::Incoming>,
    server: Arc<McpServer>,
    events: tokio::sync::broadcast::Sender<String>,
) -> Result<
    hyper::Response<http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>>,
    std::convert::Infallible,
> {
    use std::convert::Infallible;

    use bytes::Bytes;
    use http::StatusCode;
    use http_body_util::{BodyExt, Full, StreamBody};
    use hyper::body::Frame;
    use tokio_stream::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    let method = request.method().clone();
    let path = request.uri().path().to_string();

    if method == http::Method::GET && path == "/sse" {
        let endpoint_event = tokio_stream::once(Ok::<Frame<Bytes>, Infallible>(Frame::data(
            Bytes::from("event: endpoint\ndata: /message\n\n"),
        )));
        let data_stream =
            BroadcastStream::new(events.subscribe()).filter_map(|event| match event {
                Ok(message) => {
                    let payload = format!("data: {message}\n\n");
                    Some(Ok::<Frame<Bytes>, Infallible>(Frame::data(Bytes::from(
                        payload,
                    ))))
                }
                Err(_) => None,
            });
        let stream = endpoint_event.chain(data_stream);
        let body = StreamBody::new(stream).boxed();
        let response = hyper::Response::builder()
            .status(StatusCode::OK)
            .header(http::header::CONTENT_TYPE, "text/event-stream")
            .header(http::header::CACHE_CONTROL, "no-cache")
            .header(http::header::CONNECTION, "keep-alive")
            .body(body)
            .unwrap_or_else(|_| {
                hyper::Response::new(Full::new(Bytes::from_static(b"failed")).boxed())
            });
        return Ok(response);
    }

    if method == http::Method::POST && path == "/message" {
        let body = match request.into_body().collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(error) => {
                return Ok(json_response(
                    StatusCode::BAD_REQUEST,
                    crate::jsonrpc_error(
                        serde_json::Value::Null,
                        -32700,
                        &format!("read body failed: {error}"),
                    ),
                ));
            }
        };

        let message = match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(value) => value,
            Err(error) => {
                return Ok(json_response(
                    StatusCode::BAD_REQUEST,
                    crate::jsonrpc_error(
                        serde_json::Value::Null,
                        -32700,
                        &format!("parse error: {error}"),
                    ),
                ));
            }
        };

        if let Some(response) = server.handle_message(message).await
            && let Ok(encoded) = serde_json::to_string(&response)
            && events.send(encoded).is_err()
        {
            tracing::warn!("SSE broadcast channel full, message dropped");
        }

        return Ok(json_response(
            StatusCode::ACCEPTED,
            serde_json::json!({ "ok": true }),
        ));
    }

    if method == http::Method::GET && path == "/health" {
        return Ok(json_response(
            StatusCode::OK,
            serde_json::json!({ "ok": true }),
        ));
    }

    Ok(json_response(
        StatusCode::NOT_FOUND,
        serde_json::json!({ "error": "not found" }),
    ))
}

#[cfg(feature = "sse")]
fn json_response(
    status: http::StatusCode,
    value: serde_json::Value,
) -> hyper::Response<http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>> {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};

    let payload = serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec());
    hyper::Response::builder()
        .status(status)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(payload)).boxed())
        .unwrap_or_else(|_| hyper::Response::new(Full::new(Bytes::from_static(b"{}")).boxed()))
}
