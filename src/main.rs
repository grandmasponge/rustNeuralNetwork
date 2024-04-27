use std::ops::{Add, Mul, Sub};
use std::f64::consts::E;

use ndarray::{Array1, Array2, arr2};
use rand::{thread_rng, Rng};

trait Layer {
    fn calculate_foward_pass(&self, input: Array2<f64>) ->Array2<f64>;
}


struct HiddenLayer {
    neurons: usize,
    weights: Array2<f64>,
    biases: Vec<f64>,
    input_size: usize,
}

impl  HiddenLayer {
    fn new(neruons: usize, input_size: usize) ->Self {
        let weights = Self::generate_weights(input_size, neruons);
        let biases = Self::generate_biases(neruons);

        Self { neurons: neruons , weights, biases, input_size  }

    }

    fn generate_weights(input_size: usize, neurons: usize) -> Array2<f64>  {

        let mut weights = Array2::<f64>::zeros((neurons, input_size));

        for mut shape in weights.outer_iter_mut() {
           for mut row in shape.rows_mut() {
                for mut data in row.iter_mut() {
                    *data = thread_rng().gen_range(0.0..1.0); // change so that the  new weights amount is equal to the neurons amount from before that

                }

           }
        }

        return weights;
 }

    fn generate_biases(neruons: usize) ->Vec<f64> {
        let mut biases: Vec<f64> = Vec::new();
        for i in 0..neruons {
            let bias = thread_rng().gen_range(0.0..1.0);
            biases.push(bias);
        }
        biases
    }
}


impl Layer for HiddenLayer {
    fn calculate_foward_pass(&self, input: Array2<f64>) -> Array2<f64> {
        //multply inputs by weights and add biases

        let mut output = self.weights
        .dot(&input);
        let mut index = 0;

        //add biasVec<f64>
        for mut number in output.iter_mut() {
            *number = *number + self.biases[index];
        }
        output.iter_mut()
        .map(|x| 1.0 / (1.0 + E.powf(-x.clone().to_owned())));

        return output;
    }

}

struct Network {
    layers: Vec<HiddenLayer>,
    inputLength: usize
}

impl Network {
    fn new(layer_length: usize, neurons: usize, input_length: usize) -> Self {
        let mut layers: Vec<HiddenLayer> = vec![];
        for i in 0..layer_length{
            let x =  HiddenLayer::new(neurons, input_length);
            layers.push(x);
        }
        Self{
            layers,
            inputLength: input_length
        }
    }

    fn train(&self,input: Array2<f64>) -> Array2<f64> {
        let mut output = self.layers[0].calculate_foward_pass(input);
        for i in 1..self.layers.len()  {
            output = self.layers[i].calculate_foward_pass(output);
        }
        return output;
    }

    fn backwards_propagation(&self, targets: Array1<f64>, inputs: Array2<f64>) {

        let error = targets.sub(&inputs);
        let mut gradients = inputs.map(|x| x * (1.0 - x));

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let weights = layer.weights.clone();
            let biases = layer.weights.clone();

            // do somthing with backpropergation here

        }


    }

}





fn main() {

    let network = Network::new(1, 3, 2); //neurons 3 length 2 so 2x3 matrix
    let inputs = arr2(&[[1.0],
                        [0.0]]); // two inputs 1 in length so
    let output = network.train(inputs);
    println!("{output:?}");


}
