use std::sync::OnceLock;

use llm_core::{FrameworkError, Result};
#[cfg(not(feature = "claude-code-emulation"))]
use reqwest;
#[cfg(feature = "claude-code-emulation")]
use rquest::{
    AlpnProtos, AlpsProtos, CertCompressionAlgorithm, EmulationProvider, ExtensionType,
    Http1Builder, Http1Config, Http2Builder, Http2Config, Priority, PseudoOrder::*,
    SettingsOrder::*, SslCurve, StreamDependency, StreamId, TlsConfig, TlsVersion,
};

use crate::descriptor::PROVIDER_ID;

#[cfg(feature = "claude-code-emulation")]
type HttpClient = rquest::Client;
#[cfg(not(feature = "claude-code-emulation"))]
type HttpClient = reqwest::Client;

#[cfg(feature = "claude-code-emulation")]
type HttpError = rquest::Error;
#[cfg(not(feature = "claude-code-emulation"))]
type HttpError = reqwest::Error;

static ANTHROPIC_AUTH_HTTP: OnceLock<std::result::Result<HttpClient, String>> = OnceLock::new();
static ANTHROPIC_RUNTIME_HTTP: OnceLock<std::result::Result<HttpClient, String>> = OnceLock::new();

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_CURVES: &[SslCurve] = &[
    SslCurve::X25519,
    SslCurve::SECP256R1,
    SslCurve::SECP384R1,
    SslCurve::SECP521R1,
    SslCurve::FFDHE2048,
    SslCurve::FFDHE3072,
];

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_CIPHER_LIST: &str = concat!(
    "TLS_AES_128_GCM_SHA256:",
    "TLS_CHACHA20_POLY1305_SHA256:",
    "TLS_AES_256_GCM_SHA384:",
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:",
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:",
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:",
    "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA:",
    "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA:",
    "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:",
    "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:",
    "TLS_RSA_WITH_AES_128_GCM_SHA256:",
    "TLS_RSA_WITH_AES_256_GCM_SHA384:",
    "TLS_RSA_WITH_AES_128_CBC_SHA:",
    "TLS_RSA_WITH_AES_256_CBC_SHA"
);

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_SIGALGS_LIST: &str = concat!(
    "ecdsa_secp256r1_sha256:",
    "ecdsa_secp384r1_sha384:",
    "ecdsa_secp521r1_sha512:",
    "rsa_pss_rsae_sha256:",
    "rsa_pss_rsae_sha384:",
    "rsa_pss_rsae_sha512:",
    "rsa_pkcs1_sha256:",
    "rsa_pkcs1_sha384:",
    "rsa_pkcs1_sha512:",
    "ecdsa_sha1:",
    "rsa_pkcs1_sha1"
);

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_CERT_COMPRESSION: &[CertCompressionAlgorithm] = &[
    CertCompressionAlgorithm::Zlib,
    CertCompressionAlgorithm::Brotli,
    CertCompressionAlgorithm::Zstd,
];

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_DELEGATED_CREDENTIALS: &str = concat!(
    "ecdsa_secp256r1_sha256:",
    "ecdsa_secp384r1_sha384:",
    "ecdsa_secp521r1_sha512:",
    "ecdsa_sha1"
);

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_RECORD_SIZE_LIMIT: u16 = 0x4001;

#[cfg(feature = "claude-code-emulation")]
const FIREFOX_EXTENSION_PERMUTATION_INDICES: &[u8] = &{
    const EXTENSIONS: &[ExtensionType] = &[
        ExtensionType::SERVER_NAME,
        ExtensionType::EXTENDED_MASTER_SECRET,
        ExtensionType::RENEGOTIATE,
        ExtensionType::SUPPORTED_GROUPS,
        ExtensionType::EC_POINT_FORMATS,
        ExtensionType::SESSION_TICKET,
        ExtensionType::APPLICATION_LAYER_PROTOCOL_NEGOTIATION,
        ExtensionType::STATUS_REQUEST,
        ExtensionType::DELEGATED_CREDENTIAL,
        ExtensionType::KEY_SHARE,
        ExtensionType::SUPPORTED_VERSIONS,
        ExtensionType::SIGNATURE_ALGORITHMS,
        ExtensionType::PSK_KEY_EXCHANGE_MODES,
        ExtensionType::RECORD_SIZE_LIMIT,
        ExtensionType::CERT_COMPRESSION,
        ExtensionType::ENCRYPTED_CLIENT_HELLO,
    ];

    let mut indices = [0u8; EXTENSIONS.len()];
    let mut index = 0;
    while index < EXTENSIONS.len() {
        if let Some(extension_index) = ExtensionType::index_of(EXTENSIONS[index]) {
            indices[index] = extension_index as u8;
        }
        index += 1;
    }

    indices
};

#[cfg(feature = "claude-code-emulation")]
const CHROME_CURVES: &[SslCurve] = &[SslCurve::X25519, SslCurve::SECP256R1, SslCurve::SECP384R1];

#[cfg(feature = "claude-code-emulation")]
const CHROME_CIPHER_LIST: &str = concat!(
    "TLS_AES_128_GCM_SHA256:",
    "TLS_AES_256_GCM_SHA384:",
    "TLS_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256:",
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:",
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:",
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:",
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256:",
    "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:",
    "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:",
    "TLS_RSA_WITH_AES_128_GCM_SHA256:",
    "TLS_RSA_WITH_AES_256_GCM_SHA384:",
    "TLS_RSA_WITH_AES_128_CBC_SHA:",
    "TLS_RSA_WITH_AES_256_CBC_SHA"
);

#[cfg(feature = "claude-code-emulation")]
const CHROME_SIGALGS_LIST: &str = concat!(
    "ecdsa_secp256r1_sha256:",
    "rsa_pss_rsae_sha256:",
    "rsa_pkcs1_sha256:",
    "ecdsa_secp384r1_sha384:",
    "rsa_pss_rsae_sha384:",
    "rsa_pkcs1_sha384:",
    "rsa_pss_rsae_sha512:",
    "rsa_pkcs1_sha512"
);

#[cfg(feature = "claude-code-emulation")]
const CHROME_CERT_COMPRESSION: &[CertCompressionAlgorithm] = &[CertCompressionAlgorithm::Brotli];

fn provider_error(message: impl Into<String>) -> FrameworkError {
    FrameworkError::provider(PROVIDER_ID.clone(), message.into())
}

#[cfg(feature = "claude-code-emulation")]
fn firefox_auth_emulation() -> EmulationProvider {
    let tls = TlsConfig::builder()
        .curves(FIREFOX_CURVES)
        .cipher_list(FIREFOX_CIPHER_LIST)
        .sigalgs_list(FIREFOX_SIGALGS_LIST)
        .delegated_credentials(FIREFOX_DELEGATED_CREDENTIALS)
        .cert_compression_algorithm(FIREFOX_CERT_COMPRESSION)
        .record_size_limit(FIREFOX_RECORD_SIZE_LIMIT)
        .pre_shared_key(true)
        .enable_ech_grease(true)
        .alpn_protos(AlpnProtos::ALL)
        .alps_protos(AlpsProtos::HTTP2)
        .min_tls_version(TlsVersion::TLS_1_0)
        .max_tls_version(TlsVersion::TLS_1_3)
        .random_aes_hw_override(true)
        .extension_permutation_indices(FIREFOX_EXTENSION_PERMUTATION_INDICES)
        .build();

    let http1 = Http1Config::builder()
        .allow_obsolete_multiline_headers_in_responses(true)
        .max_headers(100)
        .build();

    let http2 = Http2Config::builder()
        .initial_stream_id(15)
        .header_table_size(65_536)
        .initial_stream_window_size(131_072)
        .max_frame_size(16_384)
        .initial_connection_window_size(12_517_377 + 65_535)
        .headers_priority(StreamDependency::new(StreamId::from(13), 41, false))
        .headers_pseudo_order([Method, Scheme, Authority, Path])
        .settings_order([
            HeaderTableSize,
            EnablePush,
            MaxConcurrentStreams,
            InitialWindowSize,
            MaxFrameSize,
            MaxHeaderListSize,
            UnknownSetting8,
            UnknownSetting9,
        ])
        .priority(vec![
            Priority::new(
                StreamId::from(3),
                StreamDependency::new(StreamId::zero(), 200, false),
            ),
            Priority::new(
                StreamId::from(5),
                StreamDependency::new(StreamId::zero(), 100, false),
            ),
            Priority::new(
                StreamId::from(7),
                StreamDependency::new(StreamId::zero(), 0, false),
            ),
            Priority::new(
                StreamId::from(9),
                StreamDependency::new(StreamId::from(7), 0, false),
            ),
            Priority::new(
                StreamId::from(11),
                StreamDependency::new(StreamId::from(3), 0, false),
            ),
            Priority::new(
                StreamId::from(13),
                StreamDependency::new(StreamId::zero(), 240, false),
            ),
        ])
        .build();

    EmulationProvider::builder()
        .tls_config(tls)
        .http1_config(http1)
        .http2_config(http2)
        .build()
}

#[cfg(feature = "claude-code-emulation")]
fn build_auth_http_client() -> std::result::Result<HttpClient, HttpError> {
    rquest::Client::builder()
        .emulation(firefox_auth_emulation())
        .http1(firefox_http1_configuration)
        .http2(firefox_http2_configuration)
        .build()
}

#[cfg(not(feature = "claude-code-emulation"))]
fn build_auth_http_client() -> std::result::Result<HttpClient, HttpError> {
    reqwest::Client::builder().build()
}

#[cfg(feature = "claude-code-emulation")]
fn firefox_http1_configuration(mut builder: Http1Builder<'_>) {
    builder.title_case_headers(true);
}

#[cfg(feature = "claude-code-emulation")]
fn firefox_http2_configuration(mut builder: Http2Builder<'_>) {
    builder.unknown_setting8(true);
}

#[cfg(feature = "claude-code-emulation")]
fn chrome_runtime_emulation() -> EmulationProvider {
    let tls = TlsConfig::builder()
        .curves(CHROME_CURVES)
        .cipher_list(CHROME_CIPHER_LIST)
        .sigalgs_list(CHROME_SIGALGS_LIST)
        .cert_compression_algorithm(CHROME_CERT_COMPRESSION)
        .alpn_protos(AlpnProtos::ALL)
        .alps_protos(AlpsProtos::HTTP2)
        .min_tls_version(TlsVersion::TLS_1_2)
        .max_tls_version(TlsVersion::TLS_1_3)
        .enable_ech_grease(true)
        .permute_extensions(true)
        .grease_enabled(true)
        .enable_ocsp_stapling(true)
        .enable_signed_cert_timestamps(true)
        .build();

    let http2 = Http2Config::builder()
        .initial_stream_window_size(6_291_456)
        .initial_connection_window_size(15_728_640)
        .max_header_list_size(262_144)
        .header_table_size(65_536)
        .enable_push(false)
        .build();

    EmulationProvider::builder()
        .tls_config(tls)
        .http2_config(http2)
        .build()
}

#[cfg(feature = "claude-code-emulation")]
fn build_runtime_http_client() -> std::result::Result<HttpClient, HttpError> {
    rquest::Client::builder()
        .emulation(chrome_runtime_emulation())
        .build()
}

#[cfg(not(feature = "claude-code-emulation"))]
fn build_runtime_http_client() -> std::result::Result<HttpClient, HttpError> {
    reqwest::Client::builder().build()
}

pub(crate) fn anthropic_auth_http() -> Result<&'static HttpClient> {
    match ANTHROPIC_AUTH_HTTP.get_or_init(|| build_auth_http_client().map_err(|e| e.to_string())) {
        Ok(client) => Ok(client),
        Err(err) => Err(provider_error(format!(
            "failed to build Anthropic OAuth HTTP client: {err}"
        ))),
    }
}

pub(crate) fn anthropic_runtime_http() -> Result<&'static HttpClient> {
    match ANTHROPIC_RUNTIME_HTTP
        .get_or_init(|| build_runtime_http_client().map_err(|e| e.to_string()))
    {
        Ok(client) => Ok(client),
        Err(err) => Err(provider_error(format!(
            "failed to build Anthropic runtime HTTP client: {err}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_auth_http_client() {
        assert!(build_auth_http_client().is_ok());
    }

    #[test]
    fn builds_runtime_http_client() {
        assert!(build_runtime_http_client().is_ok());
    }
}
