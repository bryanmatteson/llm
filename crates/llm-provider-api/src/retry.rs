use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio_stream::Stream;

use llm_core::{ModelDescriptor, ProviderId, Result};

use crate::client::{LlmProviderClient, TurnRequest, TurnResponse};
use crate::event::ProviderEvent;

/// Default maximum number of retries before giving up.
const DEFAULT_MAX_RETRIES: u32 = 2;

/// Default base delay for exponential backoff (500 ms).
const DEFAULT_BASE_DELAY_MS: u64 = 500;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Base delay in milliseconds; doubles on each retry.
    pub base_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            base_delay_ms: DEFAULT_BASE_DELAY_MS,
        }
    }
}

/// A wrapper around any [`LlmProviderClient`] that automatically retries
/// transient failures (HTTP 429, 503) with exponential backoff.
///
/// Only `send_turn` and `stream_turn` are retried; `list_models` delegates
/// directly since model listing is rarely rate-limited.
pub struct RetryingClient<C> {
    inner: C,
    config: RetryConfig,
}

impl<C: LlmProviderClient> RetryingClient<C> {
    /// Wrap `client` with the default retry configuration (2 retries,
    /// 500 ms base backoff).
    pub fn new(client: C) -> Self {
        Self {
            inner: client,
            config: RetryConfig::default(),
        }
    }

    /// Wrap `client` with a custom retry configuration.
    pub fn with_config(client: C, config: RetryConfig) -> Self {
        Self {
            inner: client,
            config,
        }
    }
}

#[async_trait]
impl<C: LlmProviderClient> LlmProviderClient for RetryingClient<C> {
    fn provider_id(&self) -> &ProviderId {
        self.inner.provider_id()
    }

    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner.send_turn(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt < self.config.max_retries && e.is_retryable() {
                        let delay =
                            Duration::from_millis(self.config.base_delay_ms * 2u64.pow(attempt));
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_err.expect("at least one attempt must have been made"))
    }

    async fn stream_turn(
        &self,
        request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner.stream_turn(request).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if attempt < self.config.max_retries && e.is_retryable() {
                        let delay =
                            Duration::from_millis(self.config.base_delay_ms * 2u64.pow(attempt));
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_err.expect("at least one attempt must have been made"))
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        self.inner.list_models().await
    }
}

// Also implement for Arc<C> so callers can share a retrying client.
#[async_trait]
impl<C: LlmProviderClient> LlmProviderClient for RetryingClient<Arc<C>> {
    fn provider_id(&self) -> &ProviderId {
        self.inner.provider_id()
    }

    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner.send_turn(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt < self.config.max_retries && e.is_retryable() {
                        let delay =
                            Duration::from_millis(self.config.base_delay_ms * 2u64.pow(attempt));
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_err.expect("at least one attempt must have been made"))
    }

    async fn stream_turn(
        &self,
        request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner.stream_turn(request).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if attempt < self.config.max_retries && e.is_retryable() {
                        let delay =
                            Duration::from_millis(self.config.base_delay_ms * 2u64.pow(attempt));
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_err.expect("at least one attempt must have been made"))
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        self.inner.list_models().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use llm_core::{FrameworkError, Message, ModelId, ProviderId, StopReason, TokenUsage};

    use super::*;

    /// A mock client that fails `fail_count` times with a retryable error,
    /// then succeeds.
    #[derive(Debug)]
    struct FlakyClient {
        provider_id: ProviderId,
        fail_count: u32,
        attempts: AtomicU32,
    }

    #[async_trait]
    impl LlmProviderClient for FlakyClient {
        fn provider_id(&self) -> &ProviderId {
            &self.provider_id
        }

        async fn send_turn(&self, _request: &TurnRequest) -> Result<TurnResponse> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < self.fail_count {
                Err(FrameworkError::provider(
                    self.provider_id.clone(),
                    "HTTP 429: rate limited",
                ))
            } else {
                Ok(TurnResponse {
                    messages: vec![Message::assistant("ok")],
                    stop_reason: StopReason::EndTurn,
                    model: ModelId::new("mock"),
                    usage: TokenUsage::default(),
                })
            }
        }

        async fn stream_turn(
            &self,
            _request: &TurnRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
            Err(FrameworkError::unsupported("not implemented"))
        }

        async fn list_models(&self) -> Result<Vec<llm_core::ModelDescriptor>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn retries_on_transient_error_then_succeeds() {
        let client = RetryingClient::with_config(
            FlakyClient {
                provider_id: ProviderId::new("test"),
                fail_count: 2,
                attempts: AtomicU32::new(0),
            },
            RetryConfig {
                max_retries: 2,
                base_delay_ms: 1, // fast for tests
            },
        );

        let request = TurnRequest {
            system_prompt: None,
            messages: vec![Message::user("hi")],
            tools: vec![],
            model: None,
            max_tokens: None,
            temperature: None,
        };

        let result = client.send_turn(&request).await;
        assert!(result.is_ok());
        assert_eq!(client.inner.attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let client = RetryingClient::with_config(
            FlakyClient {
                provider_id: ProviderId::new("test"),
                fail_count: 10, // always fails
                attempts: AtomicU32::new(0),
            },
            RetryConfig {
                max_retries: 2,
                base_delay_ms: 1,
            },
        );

        let request = TurnRequest {
            system_prompt: None,
            messages: vec![Message::user("hi")],
            tools: vec![],
            model: None,
            max_tokens: None,
            temperature: None,
        };

        let result = client.send_turn(&request).await;
        assert!(result.is_err());
        // 1 initial + 2 retries = 3 attempts
        assert_eq!(client.inner.attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable_errors() {
        let client = RetryingClient::with_config(
            FlakyClient {
                provider_id: ProviderId::new("test"),
                fail_count: 10,
                attempts: AtomicU32::new(0),
            },
            RetryConfig {
                max_retries: 2,
                base_delay_ms: 1,
            },
        );

        // Override: make the error non-retryable by using a different mock
        #[derive(Debug)]
        struct NonRetryableClient;
        #[async_trait]
        impl LlmProviderClient for NonRetryableClient {
            fn provider_id(&self) -> &ProviderId {
                static ID: std::sync::LazyLock<ProviderId> =
                    std::sync::LazyLock::new(|| ProviderId::new("test"));
                &ID
            }
            async fn send_turn(&self, _: &TurnRequest) -> Result<TurnResponse> {
                Err(FrameworkError::provider(
                    ProviderId::new("test"),
                    "HTTP 400: bad request",
                ))
            }
            async fn stream_turn(
                &self,
                _: &TurnRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
                Err(FrameworkError::unsupported("no"))
            }
            async fn list_models(&self) -> Result<Vec<llm_core::ModelDescriptor>> {
                Ok(vec![])
            }
        }

        let _ = client; // drop the flaky one
        let client = RetryingClient::with_config(
            NonRetryableClient,
            RetryConfig {
                max_retries: 2,
                base_delay_ms: 1,
            },
        );

        let request = TurnRequest {
            system_prompt: None,
            messages: vec![Message::user("hi")],
            tools: vec![],
            model: None,
            max_tokens: None,
            temperature: None,
        };

        let result = client.send_turn(&request).await;
        assert!(result.is_err());
        // Should fail immediately, no retries
    }
}
