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
use ureq;
use std::path::Path;

static MNIST_DATA_URL: &str = "https://raw.githubusercontent.com/fgnt/mnist/master";

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
        let files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ];
        for file in &files {
            let url = format!("{}/{}", mnist_url, file);
            let out_path = format!("{}/{}", save_dir, file);
            if !Path::new(&out_path).exists() {
                println!("Downloading: {}...", file);
                download_file(&url, &out_path)?;
            } else {
                println!("File: {}", file);
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
        let labels = read_mnist_labels(&label_file).unwrap();
        let images = read_mnist_images(&image_file).unwrap();
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

/// download a file from url
fn download_file(url: &str, out_path: &str) -> io::Result<()> {
    let mut response = ureq::get(url).call().map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    if response.status() != 200 {
        return Err(io::Error::new(io::ErrorKind::Other, format!("Failed to download file: {}", response.status())));
    }
    let mut file = File::create(out_path)?;
    let mut reader = response.body_mut().as_reader();
    io::copy(&mut reader, &mut file)?;
    Ok(())
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


/// read MNIST labels
fn read_mnist_labels(file_path: &str) -> io::Result<Vec<u8>> {
    let data = read_gzip(file_path)?;
    // skip 8 bytes of header
    let labels = data[8..].to_vec();
    Ok(labels)
}

/// read MNIST images
fn read_mnist_images(file_path: &str) -> io::Result<Vec<Vec<f32>>> {
    let raw_bytes = read_gzip(file_path)?;

    // read header
    let num_images = u32::from_be_bytes(raw_bytes[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_be_bytes(raw_bytes[8..12].try_into().unwrap()) as usize;
    let num_cols = u32::from_be_bytes(raw_bytes[12..16].try_into().unwrap()) as usize;
    let image_size = num_rows * num_cols;

    let mut images = Vec::with_capacity(num_images);
    let images_raw = &raw_bytes[16..]; // header is 16 bytes

    for i in 0..num_images {
        let start = i * image_size;
        let end = start + image_size;
        let image: Vec<f32> = images_raw[start..end]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(image);
    }
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
        let save_dir = "data";
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