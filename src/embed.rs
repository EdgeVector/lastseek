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

    /// Query-side embedding. Symmetric embedders (the default here) embed a
    /// query exactly like a document. Asymmetric models — BGE among them —
    /// override this to prepend the model's query instruction; using the
    /// document path for queries compiles fine and degrades recall silently.
    fn embed_query(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        self.embed(texts)
    }
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

/// Directory name of the bge-small-en-v1.5 file set, both under
/// `~/.lastseek/models/` and under the install's `share/lastseek/models/`.
pub const BGE_MODEL_DIR_NAME: &str = "bge-small-en-v1.5";

/// Resolve the bge-small-en-v1.5 model directory. Inputs are injected so the
/// order is testable without touching the real environment.
///
/// 1. `LASTSEEK_MODEL_DIR`, when set, wins outright.
/// 2. `$HOME/.lastseek/models/bge-small-en-v1.5`, when it holds `model.onnx`.
/// 3. `<canonical exe dir>/../share/lastseek/models/bge-small-en-v1.5`, when it
///    holds `model.onnx`. This is the Homebrew layout: the formula installs the
///    binary at `Cellar/lastseek/<v>/bin/lastseek` and the model under its
///    pkgshare, `Cellar/lastseek/<v>/share/lastseek/models/...`. The exe path
///    is canonicalized first because `/opt/homebrew/bin/lastseek` is a
///    symlink into the Cellar.
/// 4. Otherwise the `$HOME` path from step 2, so the load error names the
///    documented default location.
///
/// Not gated on the `neural` feature: the resolution is pure path logic and
/// the required CI gate (which never builds `neural`) must test it.
pub fn resolve_model_dir(
    env_model_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    current_exe: Option<std::path::PathBuf>,
) -> std::path::PathBuf {
    if let Some(v) = env_model_dir.filter(|v| !v.is_empty()) {
        return v.into();
    }
    let base = home.unwrap_or_else(|| ".".into());
    let home_dir = std::path::Path::new(&base)
        .join(".lastseek")
        .join("models")
        .join(BGE_MODEL_DIR_NAME);
    if home_dir.join("model.onnx").is_file() {
        return home_dir;
    }
    if let Some(dir) = current_exe.and_then(|exe| install_share_model_dir(&exe)) {
        if dir.join("model.onnx").is_file() {
            return dir;
        }
    }
    home_dir
}

/// `<canonical exe dir>/../share/lastseek/models/bge-small-en-v1.5`, or `None`
/// when the exe path cannot be canonicalized or has no grandparent.
fn install_share_model_dir(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    let exe = std::fs::canonicalize(exe).ok()?;
    let prefix = exe.parent()?.parent()?;
    Some(
        prefix
            .join("share")
            .join("lastseek")
            .join("models")
            .join(BGE_MODEL_DIR_NAME),
    )
}

/// `bge-small-en-v1.5` via the vendored FastEmbed crate (CLS pooling,
/// asymmetric query/document encoding).
///
/// Not vector-space compatible with the predecessor's `all-MiniLM-L6-v2`
/// index — that compatibility was explicitly not required (Tom, 2026-08-07),
/// same as the MiniLM path this replaces. 384 dimensions, so the store's slab
/// layout and `rows.bin` design are unaffected by the swap.
#[cfg(feature = "neural")]
pub struct BgeSmallEmbedder {
    model: fastembed::TextEmbedding,
}

#[cfg(feature = "neural")]
impl BgeSmallEmbedder {
    pub const ID: &'static str = "fastembed/bge-small-en-v1.5";
    pub const DIMENSIONS: usize = 384;

    /// BGE is asymmetric (Tom 2026-08-07 decision doc): queries need this
    /// instruction prefix, documents do not. Getting this backwards compiles
    /// and produces plausible vectors with silently degraded recall.
    /// https://huggingface.co/BAAI/bge-small-en-v1.5
    const QUERY_INSTRUCTION: &'static str =
        "Represent this sentence for searching relevant passages: ";

    /// Where the bge-small-en-v1.5 files are read from. See
    /// [`resolve_model_dir`] for the order. Never read out of another app's
    /// `node_modules`.
    pub fn default_model_dir() -> std::path::PathBuf {
        resolve_model_dir(
            std::env::var_os("LASTSEEK_MODEL_DIR"),
            std::env::var_os("HOME"),
            std::env::current_exe().ok(),
        )
    }

    /// Loads `model.onnx` and the tokenizer file set from `model_dir`. See
    /// README.md for how to obtain the weights.
    pub fn load(model_dir: impl AsRef<std::path::Path>) -> Result<Self, LastSeekError> {
        let model_dir = model_dir.as_ref();
        let read = |name: &str| -> Result<Vec<u8>, LastSeekError> {
            std::fs::read(model_dir.join(name)).map_err(|e| {
                LastSeekError::Embedder(format!("loading {name} from {}: {e}", model_dir.display()))
            })
        };

        let tokenizer_files = fastembed::TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        // BGE uses CLS pooling, not mean pooling — MiniLM's pooling formula
        // would silently produce plausible-but-wrong vectors here (see
        // module doc). Full precision: QuantizationMode::None, not the `Q`
        // (Static) variant.
        let model = fastembed::UserDefinedEmbeddingModel::new(read("model.onnx")?, tokenizer_files)
            .with_pooling(fastembed::Pooling::Cls)
            .with_quantization(fastembed::QuantizationMode::None);

        let model = fastembed::TextEmbedding::try_new_from_user_defined(
            model,
            fastembed::InitOptionsUserDefined::new(),
        )
        .map_err(|e| LastSeekError::Embedder(format!("loading fastembed model: {e}")))?;

        Ok(Self { model })
    }

    fn embed_raw(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        let vectors = self
            .model
            .embed(texts.to_vec(), None)
            .map_err(|e| LastSeekError::Embedder(format!("fastembed inference: {e}")))?;
        for v in &vectors {
            if v.len() != Self::DIMENSIONS {
                return Err(LastSeekError::DimensionMismatch {
                    expected: Self::DIMENSIONS,
                    got: v.len(),
                });
            }
        }
        Ok(vectors
            .into_iter()
            .map(|mut v| {
                crate::index::normalize(&mut v);
                v
            })
            .collect())
    }
}

#[cfg(feature = "neural")]
impl Embedder for BgeSmallEmbedder {
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
        self.embed_raw(texts)
    }

    fn embed_query(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", Self::QUERY_INSTRUCTION))
            .collect();
        self.embed_raw(&prefixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn touch_model(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.onnx"), b"stub").unwrap();
    }

    /// A Homebrew-shaped tree: `Cellar/lastseek/0.1.1/bin/lastseek` plus a
    /// `bin/lastseek` symlink into it, like `/opt/homebrew/bin/lastseek`.
    /// Returns (symlinked exe, pkgshare model dir).
    fn brew_layout(root: &Path) -> (PathBuf, PathBuf) {
        let keg = root.join("Cellar").join("lastseek").join("0.1.1");
        std::fs::create_dir_all(keg.join("bin")).unwrap();
        let real_exe = keg.join("bin").join("lastseek");
        std::fs::write(&real_exe, b"").unwrap();
        std::fs::create_dir_all(root.join("bin")).unwrap();
        let link = root.join("bin").join("lastseek");
        std::os::unix::fs::symlink(&real_exe, &link).unwrap();
        let share = keg
            .join("share")
            .join("lastseek")
            .join("models")
            .join(BGE_MODEL_DIR_NAME);
        (link, share)
    }

    #[test]
    fn env_override_wins_over_everything() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path().join("home");
        touch_model(&home.join(".lastseek/models").join(BGE_MODEL_DIR_NAME));
        let (exe, share) = brew_layout(&t.path().join("brew"));
        touch_model(&share);
        let got = resolve_model_dir(
            Some("/custom/model".into()),
            Some(home.into_os_string()),
            Some(exe),
        );
        assert_eq!(got, PathBuf::from("/custom/model"));
    }

    #[test]
    fn empty_env_override_is_ignored() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path().join("home");
        let got = resolve_model_dir(Some("".into()), Some(home.clone().into_os_string()), None);
        assert_eq!(got, home.join(".lastseek/models").join(BGE_MODEL_DIR_NAME));
    }

    #[test]
    fn home_model_wins_over_install_share() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path().join("home");
        let home_dir = home.join(".lastseek/models").join(BGE_MODEL_DIR_NAME);
        touch_model(&home_dir);
        let (exe, share) = brew_layout(&t.path().join("brew"));
        touch_model(&share);
        let got = resolve_model_dir(None, Some(home.into_os_string()), Some(exe));
        assert_eq!(got, home_dir);
    }

    #[test]
    fn falls_back_to_homebrew_pkgshare_through_the_bin_symlink() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path().join("home");
        // A HOME dir that exists but lacks model.onnx must not stop the fallback.
        std::fs::create_dir_all(home.join(".lastseek/models").join(BGE_MODEL_DIR_NAME)).unwrap();
        let (exe, share) = brew_layout(&t.path().join("brew"));
        touch_model(&share);
        let got = resolve_model_dir(None, Some(home.into_os_string()), Some(exe));
        assert_eq!(got, std::fs::canonicalize(&share).unwrap());
    }

    #[test]
    fn no_model_anywhere_names_the_home_default() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path().join("home");
        let (exe, _share) = brew_layout(&t.path().join("brew"));
        let got = resolve_model_dir(None, Some(home.clone().into_os_string()), Some(exe));
        assert_eq!(got, home.join(".lastseek/models").join(BGE_MODEL_DIR_NAME));
    }

    #[test]
    fn missing_exe_path_names_the_home_default() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path().join("home");
        let got = resolve_model_dir(
            None,
            Some(home.clone().into_os_string()),
            Some(t.path().join("does-not-exist")),
        );
        assert_eq!(got, home.join(".lastseek/models").join(BGE_MODEL_DIR_NAME));
    }
}
