
let numMiddleNeurons = 3;
let numMiddleLayers = 1;


const svg = document.getElementById('neural-network');
const neuronRadius = 20;
const layerDistance = 150;
const height = svg.getAttribute('height');

let numNeurons = 2;
let numLayers = 3;


// Function to clear the SVG
function clearSVG() {
    while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
    }
}



// Function to render the neural network
function renderNetwork() {
    clearSVG();

    const colors = ['#123693', 'rgb(222, 244, 254)'];

        
    // Draw connections first to make them behind the neurons
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
        const fromLayer = layers[layerIndex];
        const toLayer = layers[layerIndex + 1];

        for (let i = 0; i < fromLayer.neurons; i++) {
            for (let j = 0; j < toLayer.neurons; j++) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('class', 'connection');
                line.setAttribute('x1', fromLayer.x);
                line.setAttribute('y1', (height / (fromLayer.neurons + 1)) * (i + 1));
                line.setAttribute('x2', toLayer.x);
                line.setAttribute('y2', (height / (toLayer.neurons + 1)) * (j + 1));
                svg.appendChild(line);
            }
        }
    }

    // Draw neurons
    layers.forEach((layer, layerIndex) => {
        const color = colors[layerIndex % colors.length];
        for (let i = 0; i < layer.neurons; i++) {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', layer.x);
            circle.setAttribute('class', `neuron layer-${layerIndex}`);
            circle.setAttribute('cx', layer.x);
            circle.setAttribute('cy', (height / (layer.neurons + 1)) * (i + 1));
            circle.setAttribute('r', neuronRadius);
            circle.setAttribute('fill', color);

            svg.appendChild(circle);
        }
    });
}
function createLayerConfig() {
    let layers = [];
    const totalWidth = 600;
    const numTotalLayers = numMiddleLayers + 2; // +2 for input and output layers
    const xSpacing = totalWidth / (numTotalLayers + 1);

    for (let i = 0; i < numTotalLayers; i++) {
        const neurons = (i === 0 || i === numTotalLayers - 1) ? 2 : numMiddleNeurons;
        layers.push({ neurons: neurons, x: (i + 1) * xSpacing });
    }
    
    return layers;
}



let layers = createLayerConfig();


// Initial rendering
renderNetwork();


// Handle Neuron Buttons
const decreaseNeuronsButton = document.getElementById('decrease-neurons');
const increaseNeuronsButton = document.getElementById('increase-neurons');

decreaseNeuronsButton.addEventListener('click', () => {
    if (numMiddleNeurons > 3) {
        numMiddleNeurons -= 1;
        layers = createLayerConfig();
        renderNetwork();
    }
});

increaseNeuronsButton.addEventListener('click', () => {
    numMiddleNeurons += 1;
    layers = createLayerConfig();
    renderNetwork();
});

// Handle Layer Buttons
const decreaseLayersButton = document.getElementById('decrease-layers');
const increaseLayersButton = document.getElementById('increase-layers');
const beginTrainButton = document.getElementById('train');

decreaseLayersButton.addEventListener('click', () => {
    if (numMiddleLayers > 1) {
        numMiddleLayers -= 1;
        layers = createLayerConfig();
        renderNetwork();
    }
});

increaseLayersButton.addEventListener('click', () => {
    numMiddleLayers += 1;
    layers = createLayerConfig();
    renderNetwork();
});

trainingTextElem=document.getElementById('training-status');
epochTextElem=document.getElementById('epoch');
beginTrainButton.addEventListener('click', () => {
    let i=1;
    trainingTextElem.innerHTML="Training";
    epochTextElem.innerHTML="Epoch: "+0;
    setInterval(()=>{
        trainingTextElem.innerHTML="Training" + ".".repeat(i%4);
        epochTextElem.innerHTML="Epoch: "+Math.floor(i/2);
        i++;
    },300);

});





const draggableWindow = document.getElementById('draggable-window');
let isDragging = false;
let offsetX, offsetY;

draggableWindow.addEventListener('mousedown', (e) => {
    isDragging = true;
    e.preventDefault();
    offsetX = e.clientX - draggableWindow.getBoundingClientRect().left;
    offsetY = e.clientY - draggableWindow.getBoundingClientRect().top;
});

window.addEventListener('mouseup', () => {
    isDragging = false;
});

window.addEventListener('mousemove', (e) => {
    if (isDragging) {
        draggableWindow.style.left = (e.clientX - offsetX) + 'px';
        draggableWindow.style.top = (e.clientY - offsetY) + 'px';
    }
});
