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
}

function trainModel(net, x, y, stepSize, epochs) {
    for (let epoch = 0; epoch < epochs; epoch++) {
        let loss = new Val(0.0);

        // Zero out the gradients before each backpropagation run
        net.zeroGrads();

        for (let i = 0; i < x.length; i++) {
            let row = x[i];
            let pred = net.forward(row);
            let error = pred[0].add(new Val(-y[i]));
            loss = loss.add(error.multiply(error));
        }

        loss.backpropagate();

        for (let param of net.parameters()) {
            param.val -= stepSize * param.grad;
        }

        console.log("Epoch:", epoch + 1, " Loss:", loss.val);
    }
}

function printPredictions(net, x, y) {
    for (let i = 0; i < x.length; i++) {
        let row = x[i];
        let pred = net.forward(row);
        console.log("Prediction:", pred[0].val, "Actual:", y[i]);
    }
}

// Exporting classes and functions to be accessible in other files
// export { Val, Neuron, Layer, MLP, trainModel, printPredictions };
// Generate Sample Data
x=[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8]];
y=[0,1,2,3,4,5,6,7,8];

// Create the MLP
// let net = new MLP(input_size, hidden_layers, output_size, activations);
let net = new MLP(4, [5,5], 4, ['tanh','tanh','identity']);
console.log("Parameters: " + net.parameters().length);

// Training Configuration
const stepSize = 0.005;
const epochs = 100;

// Train the Model
trainModel(net, x, y, stepSize, epochs);

// Print Predictions
printPredictions(net, x, y);

// Test Prediction
let pred = net.forward([9,9]);
console.log("Train Network:", pred[0].val);
