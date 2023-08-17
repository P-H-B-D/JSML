class Val {
    constructor(val, children = [], operation = null, label = String(Math.floor(Math.random() * 10000))) {
        this.val = val;
        this.prev = new Set(children);
        this.operation = operation;
        this.label = label;
        this.grad = 0.0;
        this.backward = () => null;
    }

    add(other) {
        const out = new Val(this.val + other.val, [this, other], "+");
        const backward = () => {
            this.grad += 1 * out.grad;
            other.grad += 1 * out.grad;
        };
        out.backward = backward;
        return out;
    }

    multiply(other) {
        const out = new Val(this.val * other.val, [this, other], "*");
        const backward = () => {
            this.grad += other.val * out.grad;
            other.grad += this.val * out.grad;
        };
        out.backward = backward;
        return out;
    }

    tanh() {

        let x = this.val;
        const out = new Val(Math.tanh(x), [this], "tanh");
        const backward = () => {
            let y = out.val;
            this.grad += (1.0 - y * y) * out.grad;
        }
        out.backward = backward;
        return out;

    }

    identity() {

        let x = this.val;
        const out = new Val(x, [this], "identity");
        const backward = () => {
            this.grad += 1.0 * out.grad;
        }
        out.backward = backward;
        return out;

        
    }

    backpropagate() {

        //build topological order
        const nodes = new Set();
        function build(v) {
            if (!nodes.has(v)) {
                nodes.add(v);
                for (let child of v.prev) {
                    build(child);
                }
            }
        }

        build(this);

        this.grad = 1.0;
        //go one variable at a time and apply the chain rule to get its gradient
        for (let v of Array.from(nodes)) {
            v.backward();
        }
    }
    zeroGrad() {
        this.grad = 0.0;
    }
}


class Neuron {
    constructor(nInputs, activation='tanh') {
        this.weights = Array.from({ length: nInputs }, () => new Val(Math.random() * 2 - 1));
        this.bias = new Val(Math.random() * 2 - 1);
        this.activation = activation;
    }

    forward(inputs) {
        if (typeof (inputs[0]) == "number") {
            this.inputs = inputs.map((oldval, idx) => {
                return new Val(oldval);
            });
        }
        else {
            this.inputs = inputs;
        }

        let dotProduct = this.inputs.map((oldval, idx) => {
            let act = this.weights[idx].multiply(oldval);
            return act;
        });
        
        dotProduct = dotProduct.reduce((a, b) => a.add(b));
        const z = dotProduct.add(this.bias);
        
        let out;
        switch (this.activation) {
            case 'tanh':
                out = z.tanh();
                break;
            case 'identity':
                out = z.identity();
                break;

            default:
                throw new Error('Unknown activation function: ' + this.activation);
        }
        return out;

    }
    parameters() {
        return this.weights.concat(this.bias);
    }
}

class Layer {
    constructor(nInputs, nNeurons, activation='tanh') {
        this.neurons = Array.from({ length: nNeurons }, () => new Neuron(nInputs, activation));
    }

    forward(inputs) {
        let out = [];
        for (let neuron of this.neurons) {
            out.push(neuron.forward(inputs));
        }
        return out;
    }
    parameters() {
        let out = [];
        for (let neuron of this.neurons) {
            out = out.concat(neuron.parameters());
        }
        return out;
    }
}

class MLP {
    constructor(nInputs, hiddenLayers, nOutputs, activations) {
        this.layers = [];

        // Input validation: Check if the number of activations matches the number of layers
        if (activations.length !== hiddenLayers.length + 1) {
            throw new Error('Number of activations must be equal to the number of hidden layers plus 1 (for the output layer)');
        }

        let prevLayerNeuronCount = nInputs;

        // Construct hidden layers
        for (let i = 0; i < hiddenLayers.length; i++) {
            let nNeurons = hiddenLayers[i];
            let activation = activations[i];
            this.layers.push(new Layer(prevLayerNeuronCount, nNeurons, activation));
            prevLayerNeuronCount = nNeurons;
        }

        // Add output layer
        let outputActivation = activations[activations.length - 1];
        this.layers.push(new Layer(prevLayerNeuronCount, nOutputs, outputActivation));
    }

    forward(inputs) {
        let output = inputs;
        for (let layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }
    parameters() {
        let params = [];
        for (let layer of this.layers) {
            params = params.concat(layer.parameters());
        }
        return params;
    }
    zeroGrads() {
        for (let param of this.parameters()) {
            param.zeroGrad();
        }
    }
    softmax(outputs) {
        const expValues = outputs.map(val => Math.exp(val.val));
        const sumExp = expValues.reduce((sum, expVal) => sum + expVal, 0);
        const softmaxProbs = expValues.map(expVal => new Val(expVal / sumExp));
        return softmaxProbs;
    }
    softmaxCrossEntropyWithLogits(logits, labels) {
        const expValues = logits.map(val => Math.exp(val.val));
        const sumExp = expValues.reduce((sum, expVal) => sum + expVal, 0);
        const softmaxProbs = expValues.map(expVal => new Val(expVal / sumExp));

        let loss = new Val(0.0);
        for (let i = 0; i < softmaxProbs.length; i++) {
            loss = loss.add(new Val(-labels[i] * Math.log(softmaxProbs[i].val)));
        }

        const backward = () => {
            for (let i = 0; i < logits.length; i++) {
                logits[i].grad += (softmaxProbs[i].val - labels[i]);
            }
        }
        
        loss.backward = backward;
        return loss;
    }
}

function trainModel(net, x, y, stepSize, epochs) {
    const numDataPoints = x.length;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        // Randomly shuffle the data order for this epoch
        const shuffledIndices = Array.from({ length: numDataPoints }, (_, i) => i);
        for (let i = shuffledIndices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];
        }

        let loss = new Val(0.0);

        // Zero out the gradients before each backpropagation run
        net.zeroGrads();

        for (let i = 0; i < numDataPoints; i++) {
            const dataIndex = shuffledIndices[i];
            const row = x[dataIndex];
            const target = y[dataIndex];
            const pred = net.forward(row);
            const error = pred[0].add(new Val(-target));
            loss = loss.add(error.multiply(error));
        }

        loss.backpropagate();

        for (let param of net.parameters()) {
            param.val -= stepSize * param.grad;
        }

        console.log("Epoch:", epoch + 1, " Loss:", loss.val);
    }
}

function trainModelSoftmax(net, x, y, stepSize, epochs) {
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = new Val(0.0);

        // Zero out the gradients before each backpropagation run
        net.zeroGrads();

        for (let i = 0; i < x.length; i++) {
            let row = x[i];
            let logits = net.forward(row);
            
            let target = new Array(net.layers[net.layers.length - 1].neurons.length).fill(0);
            target[y[i]] = 1;

            let loss = net.softmaxCrossEntropyWithLogits(logits, target);
            loss.backpropagate();

            totalLoss = totalLoss.add(loss);
        }

        // Update parameters using gradients
        for (let param of net.parameters()) {
            param.val -= stepSize * param.grad;
        }

        console.log("Epoch:", epoch + 1, " Loss:", totalLoss.val);
    }
}



function printPredictions(net, x, y) {
    for (let i = 0; i < x.length; i++) {
        let row = x[i];
        let pred = net.forward(row);

        // Calculate softmax probabilities for the prediction
        let softmaxPred = net.softmax(pred);

        console.log("Input:", row, "Actual:", y[i], "Predicted Probabilities:", softmaxPred.map(prob => prob.val));
    }
}

// Sample data for softmax classification
const x = [
    [1, 2, 3, 4], // Input features
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7]
];
const y = [0, 1, 0, 1]; // Target labels

// Create the MLP
const inputSize = x[0].length;
const hiddenLayers = [5, 3]; // Number of neurons in hidden layers
const outputSize = 3; // Number of classes
const activations = ['tanh', 'tanh', 'identity']; // Activation functions for layers

const net = new MLP(inputSize, hiddenLayers, outputSize, activations);

// Training Configuration
const stepSize = 0.01;
const epochs = 50;

// Train the Model
trainModelSoftmax(net, x, y, stepSize, epochs);

// Test and print predictions
console.log("Predictions after training:");
printPredictions(net, x, y);


// Test prediction with new data
const newTestData = [[5, 6, 7, 8]]; // Input size should match the defined inputSize
const prediction = net.forward(newTestData[0]);
console.log("Predicted class probabilities:", prediction[0].val);