use mnist_reader::{MnistReader, print_image};
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
