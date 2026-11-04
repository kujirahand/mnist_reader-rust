//! # MNIST data reader
//! This module provides a reader for the MNIST dataset.
//! It downloads the dataset from GitHub if it is not already present in the specified directory.
//! It also provides methods to load the data into memory.
//! 
//! You can easily download and use the MNIST data as shown below.
//! 
//! ```rust
//! use mnist_reader::{MnistReader, print_image};
//! fn main() {
//!    // read MNIST data
//!     let mut mnist = MnistReader::new("mnist-data");
//!    // download MNIST data
//!     mnist.load().unwrap();
//!     // print the size of the data
//!     println!("Train data size: {}", mnist.train_data.len());
//!     println!("Test data size: {}", mnist.test_data.len());
//!     println!("Train labels size: {}", mnist.train_labels.len());
//!     println!("Test labels size: {}", mnist.test_labels.len());
//!     // print the first image
//!     let train_data: Vec<Vec<f32>> = mnist.train_data;
//!     println!("images[0]={:?}", train_data[0]);
//!     print_image(&train_data[0]);
//!     // print the first label
//!     let train_labels: Vec<u8> = mnist.train_labels;
//!     println!("labels[0]={:?}", train_labels[0]);
//! }
//! ```
//! 
use std::fs::{self, File};
use std::io::{self, Read};
use flate2::read::GzDecoder;
use std::path::Path;
use sha2::{Digest, Sha256};

static MNIST_DATA_URL: &str = "https://raw.githubusercontent.com/fgnt/mnist/master";

/// MNIST file names and their expected SHA-256 digests
const MNIST_FILES: [(&str, &str); 4] = [
    ("train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    ("train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    ("t10k-images-idx3-ubyte.gz", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    ("t10k-labels-idx1-ubyte.gz", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
];

/// MNIST data reader
/// This struct is used to read MNIST data from the given directory.
/// It downloads the data files if they are not already present in the directory.
/// It also provides methods to load the data into memory.
#[derive(Debug)]
pub struct MnistReader {
    pub train_labels: Vec<u8>,
    pub train_data: Vec<Vec<f32>>,
    pub test_labels: Vec<u8>,
    pub test_data: Vec<Vec<f32>>,
    pub mnist_url: String,
    pub save_dir: String,
}
impl MnistReader {
    /// create a new MnistReader
    pub fn new(save_dir: &str) -> Self {
        MnistReader {
            train_labels: Vec::new(),
            train_data: Vec::new(),
            test_labels: Vec::new(),
            test_data: Vec::new(),
            mnist_url: MNIST_DATA_URL.to_string(),
            save_dir: save_dir.to_string(),
        }
    }
    /// download MNIST data files
    pub fn download_files(save_dir: &str, mnist_url: &str) -> io::Result<()> {
        // check directory
        fs::create_dir_all(save_dir)?;
        // download files
        for (file, sha256) in &MNIST_FILES {
            let url = format!("{}/{}", mnist_url, file);
            let out_path = format!("{}/{}", save_dir, file);
            if !Path::new(&out_path).exists() {
                println!("Downloading: {}...", file);
                download_file(&url, &out_path, sha256)?;
            } else {
                println!("File: {}", file);
                verify_sha256(&out_path, sha256)?;
            }
        }
        Ok(())
    }
    /// load all MNIST data
    pub fn load(&mut self) -> io::Result<()> {
        // check directory
        Self::download_files(&self.save_dir, &self.mnist_url)?;
        // load train data
        self.load_data(true)?;
        self.load_data(false)?;
        Ok(())

    }
    /// load MNIST data
    fn load_data(&mut self, is_train: bool) -> io::Result<()> {
        let type_str = if is_train { "train" } else { "t10k" };
        let label_file = format!("{}/{}-labels-idx1-ubyte.gz", self.save_dir, type_str);
        let image_file = format!("{}/{}-images-idx3-ubyte.gz", self.save_dir, type_str);
        let labels = read_mnist_labels(&label_file)?;
        let images = read_mnist_images(&image_file)?;
        if is_train {
            self.train_labels = labels;
            self.train_data = images;
        } else {
            self.test_labels = labels;
            self.test_data = images;
        }
        Ok(())
    }

}

/// verify the SHA-256 digest of a file
fn verify_sha256(path: &str, expected: &str) -> io::Result<()> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual: String = hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect();
    if actual != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("checksum mismatch for {}: expected {}, got {}", path, expected, actual),
        ));
    }
    Ok(())
}

/// download a file from url
fn download_file(url: &str, out_path: &str, sha256: &str) -> io::Result<()> {
    let mut response = ureq::get(url).call().map_err(|e| io::Error::other(e.to_string()))?;
    if response.status() != 200 {
        return Err(io::Error::other(format!("Failed to download file: {}", response.status())));
    }
    // write to a temporary file first so an interrupted download never leaves a broken file
    let tmp_path = format!("{}.part", out_path);
    let result = (|| {
        let mut file = File::create(&tmp_path)?;
        let mut reader = response.body_mut().as_reader();
        io::copy(&mut reader, &mut file)?;
        file.sync_all()?;
        verify_sha256(&tmp_path, sha256)?;
        fs::rename(&tmp_path, out_path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}


/// ungzip a file
pub fn ungzip(in_path: &str, out_path: &str) -> io::Result<()> {
    let input = File::open(in_path)?;
    let mut output = File::create(out_path)?;
    let mut decoder = GzDecoder::new(input);
    io::copy(&mut decoder, &mut output)?;
    Ok(())
}

/// read from gzip file to memory
pub fn read_gzip(in_path: &str) -> io::Result<Vec<u8>> {
    let file = File::open(in_path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;
    Ok(buffer)
}


fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_u32_be(data: &[u8], offset: usize) -> io::Result<usize> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or_else(|| invalid("file is too short"))?
        .try_into()
        .map_err(|_| invalid("file is too short"))?;
    Ok(u32::from_be_bytes(bytes) as usize)
}

/// read MNIST labels
fn read_mnist_labels(file_path: &str) -> io::Result<Vec<u8>> {
    let data = read_gzip(file_path)?;
    if read_u32_be(&data, 0)? != 0x0000_0801 {
        return Err(invalid("invalid magic number in label file"));
    }
    let num_labels = read_u32_be(&data, 4)?;
    let labels = &data[8..];
    if labels.len() != num_labels {
        return Err(invalid("label count does not match header"));
    }
    Ok(labels.to_vec())
}

/// read MNIST images
fn read_mnist_images(file_path: &str) -> io::Result<Vec<Vec<f32>>> {
    let raw_bytes = read_gzip(file_path)?;

    // read header
    if read_u32_be(&raw_bytes, 0)? != 0x0000_0803 {
        return Err(invalid("invalid magic number in image file"));
    }
    let num_images = read_u32_be(&raw_bytes, 4)?;
    let num_rows = read_u32_be(&raw_bytes, 8)?;
    let num_cols = read_u32_be(&raw_bytes, 12)?;
    let image_size = num_rows
        .checked_mul(num_cols)
        .ok_or_else(|| invalid("invalid image size"))?;
    let images_raw = &raw_bytes[16..]; // header is 16 bytes
    let expected = num_images
        .checked_mul(image_size)
        .ok_or_else(|| invalid("invalid image count"))?;
    if image_size == 0 || images_raw.len() != expected {
        return Err(invalid("image data size does not match header"));
    }

    let images = images_raw
        .chunks_exact(image_size)
        .map(|img| img.iter().map(|&b| b as f32 / 255.0).collect())
        .collect();
    Ok(images)
}

/// print MNIST image data
pub fn print_image(image: &[f32]) {
    for row in image.chunks(28) {
        for &pixel in row {
            if pixel > 0.5 {
                print!("*");
            } else {
                print!("_");
            }
        }
        println!();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_files() {
        let save_dir = "mnist-data";
        let mut reader = MnistReader::new(save_dir);
        reader.load().unwrap();
        assert!(!reader.train_labels.is_empty());
        assert!(!reader.train_data.is_empty());
        assert!(!reader.test_labels.is_empty());
        assert!(!reader.test_data.is_empty());
        assert_eq!(reader.train_labels.len(), 60000);
        assert_eq!(reader.train_data.len(), 60000);
        assert_eq!(reader.test_labels.len(), 10000);
        assert_eq!(reader.test_data.len(), 10000);
        let train_labels = reader.train_labels.clone();
        println!("train_labels: {:?}", train_labels);
    }
}