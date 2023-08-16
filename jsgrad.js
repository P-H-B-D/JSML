const graphviz = require('graphviz');

function trace(root) {
    let nodes = new Set();
    let edges = new Set();

    function build(v) {
        if (!nodes.has(v)) {
            nodes.add(v);
            for (let child of v.prev) {
                edges.add([child, v]);
                build(child);
            }
        }
    }

    build(root);
    return { nodes, edges };
}

function drawDot(root) {
    const { nodes, edges } = trace(root);

    let g = graphviz.digraph('G');
    g.set('rankdir', 'LR');

    nodes.forEach(n => {
        let uid = n.label ? n.label : "null";
        g.addNode(uid, {
            label: `{ ${n.label ? n.label : "null"} | val ${n.val.toFixed(4)} | grad ${n.grad.toFixed(4)} }`,
            shape: 'record'
        });

        if (n.operation) {
            g.addNode(uid + n.operation, {
                label: n.operation
            });
            g.addEdge(uid + n.operation, uid);
        }
    });

    edges.forEach(([n1, n2]) => {
        g.addEdge(n1.label ? n1.label : "null", (n2.label ? n2.label : "null") + n2.operation);
    });

    return g;
}



class Val {
    constructor(val, children = [], operation = null, label = String(Math.floor(Math.random() * 10000))) {
        this.val = val;
        this.prev = new Set(children);
        this.operation = operation;
        this.label = label;
        this.grad = 0.0;
        this.backward = () => {
            return null;
        };
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
}


class Neuron {
    constructor(nInputs) {
        this.weights = Array.from({ length: nInputs }, () => new Val(Math.random() * 2 - 1));
        this.bias = new Val(Math.random() * 2 - 1);
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
        const out = dotProduct.add(this.bias).tanh();
        return out;

    }
    parameters() {
        return this.weights.concat(this.bias);
    }
}

class Layer {
    constructor(nInputs, nNeurons) {
        this.neurons = Array.from({ length: nNeurons }, () => new Neuron(nInputs));
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
    constructor(nInputs, nHidden, nOutputs) {
        this.hidden = new Layer(nInputs, nHidden);
        this.output = new Layer(nHidden, nOutputs);
    }

    forward(inputs) {
        let out = this.hidden.forward(inputs);
        out = this.output.forward(out);
        return out;
    }
    parameters() {
        let out = [];
        out = out.concat(this.hidden.parameters());
        out = out.concat(this.output.parameters());
        return out;
    }

}


x = [[3.0, -5.2, 7.1],
[1.0, 2.0, 3.0],
[2.0, 4.0, 0.0],
[1.0, 1.0, 1.0],
];

y = [1.0, -1.0, -1.0, -1.0];

let net = new MLP(3, 5, 1);
console.log("Parameters: " + net.parameters().length);

// make predictions for each row
for (let i = 0; i < x.length; i++) {
    let row = x[i];
    let pred = net.forward(row);
}

var loss = new Val(0.0);

//calculate loss
for (let i = 0; i < x.length; i++) {
    let row = x[i];
    let pred = net.forward(row);
    let error = pred[0].add(new Val(-y[i]));
    loss = loss.add(error.multiply(error));
}
loss.backpropagate();
console.log("Loss: " + loss.val)


let stepSize = 0.01;
for (i = 0; i < 50; i++) {
    //move in the opposite direction of the gradient

    for (let param of net.parameters()) {
        param.val -= stepSize * param.grad;
    }

    // make predictions for each row
    for (let i = 0; i < x.length; i++) {
        let row = x[i];
        let pred = net.forward(row);
    }

    var loss = new Val(0.0);

    //calculate loss
    for (let i = 0; i < x.length; i++) {
        let row = x[i];
        let pred = net.forward(row);
        let error = pred[0].add(new Val(-y[i]));
        loss = loss.add(error.multiply(error));
    }
    loss.backpropagate();
    console.log("Loss: " + loss.val)
}

// print the predictions
for (i = 0; i < x.length; i++) {
    let row = x[i];
    let pred = net.forward(row);
    console.log(pred[0].val, y[i]);
}

let pred = net.forward([1.0, 2.0, 3.0]);
console.log(pred[0].val);



// const graph = drawDot(loss);
// graph.render('png', 'output.png', function(err) {
//     if (err) {
//         console.error("Error rendering graph:", err);
//     } else {
//         console.log("Graph rendered and saved to output.png");
//     }
// });