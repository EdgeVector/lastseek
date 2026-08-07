//! Embedders.
//!
//! The neural embedder is deliberately behind a trait and an optional feature.
//! CI must be able to prove the store and the search semantics without
//! downloading model weights or linking an inference runtime — the predecessor
//! made the same call and it is the reason its tests stayed fast and offline.
//!
//! A deterministic embedder is a **test fixture only**. The predecessor let
//! deterministic vectors reach the production index, where 2,021 of them ended
//! up sharing a space with neural ones and being scored against them as if they
//! were comparable. `Embedder::is_neural` exists so that mixing is refusable at
//! the boundary rather than discovered later in a coverage audit.

use crate::LastSeekError;

pub trait Embedder {
    /// Identity recorded in the manifest. Vectors from different ids are not
    /// comparable and must never share a store.
    fn id(&self) -> &str;
    fn dimensions(&self) -> usize;
    /// False for fixtures. A production index refuses these.
    fn is_neural(&self) -> bool;
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError>;
}

/// Hash-based stand-in with the right shape and no semantics.
///
/// Exists so tests can exercise storage, scope and ranking offline. It is not
/// an approximation of a language model and must never be written to a
/// production store.
pub struct DeterministicEmbedder {
    dims: usize,
}

impl DeterministicEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Embedder for DeterministicEmbedder {
    fn id(&self) -> &str {
        "deterministic-test-fixture"
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn is_neural(&self) -> bool {
        false
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dims];
                // Token-level hashing, so shared words produce genuine overlap
                // and ranking tests are meaningful rather than arbitrary.
                for tok in t.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                    if tok.is_empty() {
                        continue;
                    }
                    let mut h: u64 = 1469598103934665603;
                    for b in tok.as_bytes() {
                        h ^= *b as u64;
                        h = h.wrapping_mul(1099511628211);
                    }
                    v[(h % self.dims as u64) as usize] += 1.0;
                }
                crate::index::normalize(&mut v);
                v
            })
            .collect())
    }
}

/// Full-precision `all-MiniLM-L6-v2` behind `ort` (ONNX Runtime) + `tokenizers`.
///
/// Not vector-space compatible with the predecessor's int8
/// `model_quantized.onnx` index — that compatibility was explicitly not
/// required (Tom, 2026-08-07), which is what let this use full precision and
/// start clean of the predecessor's polluted vectors instead of chasing
/// bit-for-bit parity.
#[cfg(feature = "neural")]
pub struct MiniLmEmbedder {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "neural")]
impl MiniLmEmbedder {
    pub const ID: &'static str = "all-MiniLM-L6-v2";
    pub const DIMENSIONS: usize = 384;

    /// `~/.lastseek/models/all-MiniLM-L6-v2`, overridable via
    /// `LASTSEEK_MODEL_DIR`. Never read out of another app's `node_modules`.
    pub fn default_model_dir() -> std::path::PathBuf {
        if let Ok(v) = std::env::var("LASTSEEK_MODEL_DIR") {
            return v.into();
        }
        let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        std::path::Path::new(&base)
            .join(".lastseek")
            .join("models")
            .join("all-MiniLM-L6-v2")
    }

    /// Loads `model.onnx` and `tokenizer.json` from `model_dir`.
    pub fn load(model_dir: impl AsRef<std::path::Path>) -> Result<Self, LastSeekError> {
        let model_dir = model_dir.as_ref();

        let mut tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| {
                LastSeekError::Embedder(format!(
                    "loading tokenizer.json from {}: {e}",
                    model_dir.display()
                ))
            })?;
        // The tokenizer.json shipped by sentence-transformers bakes in a
        // padding strategy — the spike saw every input padded to 128 tokens,
        // and pad tokens leak into the mean-pooled vector unless padding is
        // turned off here. Batching stays one text at a time as a result.
        tokenizer.with_padding(None);

        let session = ort::session::Session::builder()
            .map_err(|e| LastSeekError::Embedder(format!("ort session builder: {e}")))?
            .commit_from_file(model_dir.join("model.onnx"))
            .map_err(|e| {
                LastSeekError::Embedder(format!(
                    "loading model.onnx from {}: {e}",
                    model_dir.display()
                ))
            })?;

        Ok(Self { session, tokenizer })
    }
}

#[cfg(feature = "neural")]
impl Embedder for MiniLmEmbedder {
    fn id(&self) -> &str {
        Self::ID
    }

    fn dimensions(&self) -> usize {
        Self::DIMENSIONS
    }

    fn is_neural(&self) -> bool {
        true
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        texts
            .iter()
            .map(|text| {
                let encoding = self
                    .tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| LastSeekError::Embedder(format!("tokenize: {e}")))?;
                let seq_len = encoding.len();
                let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
                let mask: Vec<i64> = encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&x| x as i64)
                    .collect();
                let type_ids: Vec<i64> =
                    encoding.get_type_ids().iter().map(|&x| x as i64).collect();

                let outputs = self
                    .session
                    .run(ort::inputs! {
                        "input_ids" => ort::value::Tensor::from_array(([1, seq_len], ids))
                            .map_err(|e| LastSeekError::Embedder(format!("input_ids tensor: {e}")))?,
                        "attention_mask" => ort::value::Tensor::from_array(([1, seq_len], mask.clone()))
                            .map_err(|e| LastSeekError::Embedder(format!("attention_mask tensor: {e}")))?,
                        "token_type_ids" => ort::value::Tensor::from_array(([1, seq_len], type_ids))
                            .map_err(|e| LastSeekError::Embedder(format!("token_type_ids tensor: {e}")))?,
                    })
                    .map_err(|e| LastSeekError::Embedder(format!("inference: {e}")))?;

                let (shape, data) = outputs["last_hidden_state"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| LastSeekError::Embedder(format!("extract last_hidden_state: {e}")))?;
                let hidden = *shape.last().expect("last_hidden_state has rank >= 1") as usize;
                if hidden != Self::DIMENSIONS {
                    return Err(LastSeekError::DimensionMismatch {
                        expected: Self::DIMENSIONS,
                        got: hidden,
                    });
                }

                // Attention-mask-weighted mean pool. With padding disabled the
                // mask is all ones and this is a plain mean, but writing it
                // this way keeps the formula correct if a padded/batched path
                // is ever added.
                let mut pooled = vec![0f32; hidden];
                let mut weight = 0f32;
                for t in 0..seq_len {
                    let m = mask[t] as f32;
                    if m == 0.0 {
                        continue;
                    }
                    weight += m;
                    let row = &data[t * hidden..(t + 1) * hidden];
                    for (p, x) in pooled.iter_mut().zip(row) {
                        *p += x * m;
                    }
                }
                if weight > 0.0 {
                    for p in pooled.iter_mut() {
                        *p /= weight;
                    }
                }
                crate::index::normalize(&mut pooled);
                Ok(pooled)
            })
            .collect()
    }
}
