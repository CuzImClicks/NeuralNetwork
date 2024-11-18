
### Neural Network

### 0. base
Elapsed: 22.2500019s
44943.82 epochs/s

### 2. zs with_capacity

Elapsed: 2.2176342s
45093.10 epochs/s

### 3. activation inline

Elapsed: 2.1973911s
45508.52 epochs/s

### 4. single allocation for nabla_w and nabla_b

Elapsed: 2.0689128s
48334.57 epochs/s

### 5. stop activations from growing inside a loop

Elapsed: 843.3289ms
118577.70 epochs/s

### 6. stop allocating delta_nabla_w and delta_nabla_b inside a loop

Elapsed: 783.4655ms
127638.05 epochs/s

### 7. sigmoid prime on cost derivative in place

Elapsed: 748.3928ms
133619.67 epochs/s

---
- [ ] Files
  - [ ] Saving
  - [ ] Loading

- [ ] Training
  - [ ] Backpropagation
  - [ ] Gradient Descent
  
- [ ] Datasets
  - [ ] Loading
  - [ ] Batching
  - [ ] Shuffling


- [ ] Layers
  - [ ] TODO
- [ ] Activation Functions
- [ ] Loss Functions
- [ ] Optimizers
  - [ ] GPU?
  - [ ] Multi-threading?

---
Linear Regression
Task: 
**Predict** a continuous value based on one or more inputs. A common example is 
predicting house prices based on features like size and number of bedrooms.<br>
**Why**: It helps understand how neural networks can approximate continuous functions.<br>
**Data**: Generate a synthetic dataset where the output is a linear combination of 
inputs, possibly with some added noise.<br>

---

[Repo with good python code](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py)

[Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

[Math relating to Backpropagation](https://www.youtube.com/watch?v=tIeHLnjs5U8)

[Why naming variables is important](https://github.com/MikhailKravets/NeuroFlow/blob/master/src/lib.rs)

[Good implementation but convoluted](https://github.com/jackm321/RustNN/blob/master/src/lib.rs)

[Better implementation didnt try it yet](https://github.com/Vercaca/NN-Backpropagation/blob/master/neural_network.py#L28)

[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)

[Visualize Neural Network](https://alexlenail.me/NN-SVG/index.html)
[Visualize Tflite models](https://netron.app/)



```math
a_{n+1}=a_{n} - \gamma \nabla F(a)
```

```rust
fn main() {
  let mut n = NeuralNetwork {
    weights: vec ! [arr2( & [[-6.48660725, -6.62301231], [4.66472526, 4.68940699]]), arr2( & [[-9.6819727, -10.03086451]])],
    biases: vec ! [arr2( &[[2.59613863], [-7.27095101]]), arr2( & [[4.87653359]])],
  };
}
```
