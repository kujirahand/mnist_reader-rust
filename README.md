# MNIST Reader for Rust

This crate provides a reader for the MNIST dataset.

It downloads the dataset from GitHub if it is not already present in the specified directory. It also provides methods to load the data into memory.

You can easily download and use the MNIST data as shown below.

```rust
use crate::mnist_reader::{MnistReader, print_image};

fn main() {
    // read MNIST data
    let mut mnist = MnistReader::new("mnist-data");
    // download MNIST data
    mnist.load().unwrap();
    println!("Train data size: {}", mnist.train_data.len());
    println!("Test data size: {}", mnist.test_data.len());
    println!("Train labels size: {}", mnist.train_labels.len());
    println!("Test labels size: {}", mnist.test_labels.len());
    // print the first image
    let train_data: Vec<Vec<f32>> = mnist.train_data;
    println!("images[0]={:?}", train_data[0]);
    print_image(&train_data[0]);
    // print the first label
    let train_labels: Vec<u8> = mnist.train_labels;
    println!("labels[0]={:?}", train_labels[0]);
}
```

## Link

- [GitHub > MNIST Reader for Rust](https://github.com/kujirahand/mnist_reader-rust)

- [Crates.io > mnist_reader](https://crates.io/crates/mnist_reader)
- [Doc.rs > mnist_reader](https://docs.rs/mnist_reader/)
