use anyhow::{Context, Result, bail};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Binary,
    Json,
}

impl Format {
    fn extension(&self) -> &'static str {
        match self {
            Format::Binary => "bin",
            Format::Json => "json",
        }
    }
}

fn to_bytes<T: Serialize>(value: &T, fmt: Format) -> Result<Vec<u8>> {
    let bytes = match fmt {
        Format::Binary => bincode::serialize(value)?,
        Format::Json => serde_json::to_vec_pretty(value)?,
    };

    Ok(bytes)
}

fn from_bytes<T: for<'de> Deserialize<'de>>(bytes: &[u8], fmt: Format) -> Result<T> {
    let value = match fmt {
        Format::Binary => bincode::deserialize(bytes)?,
        Format::Json => serde_json::from_slice(bytes)?,
    };

    Ok(value)
}

pub fn save_to_file<T: Serialize>(path: impl AsRef<Path>, value: &T, fmt: Format) -> Result<()> {
    let bytes = to_bytes(value, fmt)?;
    let path = path.as_ref();
    if path.extension().is_none_or(|it| it != fmt.extension()) {
        bail!("Trying to create file `{path:?}` with wrong extension for data type `{fmt:?}`")
    }
    let mut f = File::create(path).with_context(|| format!("creating {:?}", path))?;

    f.write_all(&bytes)?;

    Ok(())
}

pub fn load_from_file<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
    fmt: Format,
) -> Result<T> {
    let bytes = fs::read(path.as_ref()).with_context(|| format!("reading {:?}", path.as_ref()))?;

    from_bytes(&bytes, fmt)
}
