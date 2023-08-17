function runMLScript() {
    // Generate Sample Data
    x=[[0,0],[0,1],[1,0],[1,1]];
    y=[0,1,1,0];
    
    // MLP Configuration
    const input_size = 2;
    const hidden_layers = [10]; // One hidden layer with 10 neurons
    const output_size = 1;
    const activations = ['tanh','tanh']; // 'tanh' for hidden layers, 'identity' for output layer
    
    // Create the MLP
    // let net = new MLP(input_size, hidden_layers, output_size, activations);
    let net = new MLP(2, [5], 2, ['tanh', 'tanh']);
    console.log("Parameters: " + net.parameters().length);
    
    // Training Configuration
    const stepSize = 0.01;
    const epochs = 100;
    
    // Train the Model
    trainModel(net, x, y, stepSize, epochs);
    
    // Print Predictions
    printPredictions(net, x, y);
    
    // Test Prediction
    let pred = net.forward([0.5, 0.3]);
    console.log("Train Network:", pred[0].val);
    
    // Display the prediction on the HTML page
    const predictionElement = document.getElementById("prediction");
    predictionElement.textContent = "Test Prediction: " + pred[0].val;
}

// Attach the runMLScript function to the button click event
const runButton = document.getElementById("runButton");
runButton.addEventListener("click", runMLScript);
