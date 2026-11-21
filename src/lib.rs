pub mod basicblock;
pub mod crypto;
pub mod dag;
pub mod onnx;
pub mod util;

use crate::util::config::Config;
use once_cell::sync::Lazy;
use std::env;
use std::fs::File;
use std::io::Read;

pub static CONFIG_FILE: Lazy<String> = Lazy::new(|| {
  if let Ok(cfg) = env::var("ZKTRANSFORMER_CONFIG") {
    return cfg;
  }
  let args: Vec<String> = env::args().collect();
  if args.len() >= 2 && std::path::Path::new(&args[1]).is_file() {
    return args[1].clone();
  }
  if std::path::Path::new("config.yaml").is_file() {
    return "config.yaml".to_string();
  }
  panic!("Usage: cargo run -- <config file> (or set ZKTRANSFORMER_CONFIG / provide config.yaml)");
});

// Define a static CONFIG that holds the loaded configuration
pub static CONFIG: Lazy<Config> = Lazy::new(|| {
  let mut file = File::open(&*CONFIG_FILE).expect("Could not open config");
  let mut contents = String::new();
  file.read_to_string(&mut contents).expect("Could not read config");

  serde_yaml::from_str(&contents).expect("Could not parse config")
});

pub static SF_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.scale_factor_log);
pub static TABLE_SIZE_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.table_size_log);
pub static TABLE_COMMIT_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.table_commit_log);
pub static SF_FLOAT: Lazy<f32> = Lazy::new(|| (1 << *SF_LOG) as f32);
pub static SF_INT: Lazy<usize> = Lazy::new(|| (1 << *SF_LOG) as usize);
// Compile-time constants based on feature flags
#[cfg(feature = "bn254")]
pub const SIGN_BIT: usize = 253;
#[cfg(feature = "bn254")]
pub const FIELD_SIZE: usize = 254;

#[cfg(feature = "bls12_381")]
pub const SIGN_BIT: usize = 254;
#[cfg(feature = "bls12_381")]
pub const FIELD_SIZE: usize = 256;

#[cfg(feature = "goldilocks")]
pub const SIGN_BIT: usize = 63;
#[cfg(feature = "goldilocks")]
pub const FIELD_SIZE: usize = 64;
