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
    return {nodes, edges};
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
    constructor(val, children = [], operation = null, label = null) {
        this.val = val;
        this.prev = new Set(children);
        this.operation = operation;
        this.label = label;
        this.grad= 0.0;
        this.backward = () => {
            return null;
        };
    }

    add(other) {
        const out= new Val(this.val + other.val, [this, other], "+");
        const backward = () => {
            this.grad += 1 * out.grad;
            other.grad += 1 * out.grad;
        };
        out.backward = backward;
        return out;
    }

    multiply(other) {
        const out= new Val(this.val * other.val, [this, other], "*");
        const backward = () => {
            this.grad += other.val * out.grad;
            other.grad += this.val * out.grad;
        };
        out.backward = backward;
        return out;
    }

    tanh() {

        let x= this.val;
        const out= new Val(Math.tanh(x), [this], "tanh");
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
        this.grad= 1.0;
        //go one variable at a time and apply the chain rule to get its gradient
        for (let v of Array.from(nodes)) {
            v.backward();
        }
    }
}


class Neuron{
    constructor(nInputs){
        this.weights= Array.from({ length: nInputs }, () => new Val(Math.random() * 2 - 1, [], null, "w"));
        this.bias= new Val (Math.random() * 2 - 1);
    }
    
    forward(inputs){
        // console.log(this.weights[0].add(this.weights[1]))
        let outArr=inputs.map((oldval,idx)=>{
            let oldValObj= new Val(oldval, [], null, "w");
            let act = this.weights[idx].multiply(oldValObj).add(this.bias);
            return act;
        });

        var sum=new Val(0);
        for (var i = 0; i < outArr.length; i++) {
            sum = sum.add(outArr[i]);
        }

        sum = sum.tanh();
        sum.backpropagate();
        return sum;


    }
}

x=[2.0,3.0];
let k = new Neuron(2);

console.log(k.forward(x));





// Forward pass
const b=new Val(6.88137, [], null, "b");
const x2= new Val(0.0, [], null, "x2");
const w2= new Val(1.0, [], null, "w2");

const x1= new Val(2.0, [], null, "x1");
const w1= new Val(-3.0, [], null, "w1");

const x2w2= x2.multiply(w2); x2w2.label= "x2*w2";
const x1w1= x1.multiply(w1); x1w1.label= "x1*w1";
const x1w1x2w2= x1w1.add(x2w2); x1w1x2w2.label= "x1w1+x2w2";
const n=x1w1x2w2.add(b); n.label= "n";
const o=n.tanh(); o.label= "o";


o.backpropagate();



const result = o;





const graph = drawDot(result);
graph.render('png', 'output.png', function(err) {
    if (err) {
        console.error("Error rendering graph:", err);
    } else {
        console.log("Graph rendered and saved to output.png");
    }
});