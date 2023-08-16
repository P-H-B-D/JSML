
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



const draggableWindow2 = document.getElementById('draggable-window2');
const canvas = document.getElementById('pixelCanvas');
const ctx = canvas.getContext('2d');
let isDragging2 = false;
let drawing = false;
let offsetX2, offsetY2;

draggableWindow2.addEventListener('mousedown', (e) => {
    isDragging2 = true;
    e.preventDefault();
    offsetX2 = e.clientX - draggableWindow2.getBoundingClientRect().left;
    offsetY2 = e.clientY - draggableWindow2.getBoundingClientRect().top;
});

window.addEventListener('mouseup', () => {
    isDragging2 = false;
});

window.addEventListener('mousemove', (e) => {
    if (isDragging2) {
        // Check if the cursor is not over the pixelCanvas
        if (e.target !== canvas) {
            draggableWindow2.style.left = (e.clientX - offsetX2) + 'px';
            draggableWindow2.style.top = (e.clientY - offsetY2) + 'px';
        }
    }
});


const draggableWindow3 = document.getElementById('draggable-window3');
let isDragging3 = false;
let offsetX3, offsetY3;

draggableWindow3.addEventListener('mousedown', (e) => {
    isDragging3 = true;
    e.preventDefault();
    offsetX3 = e.clientX - draggableWindow3.getBoundingClientRect().left;
    offsetY3 = e.clientY - draggableWindow3.getBoundingClientRect().top;
});

window.addEventListener('mouseup', () => {
    isDragging3 = false;
});

window.addEventListener('mousemove', (e) => {
    if (isDragging3) {
        draggableWindow3.style.left = (e.clientX - offsetX3) + 'px';
        draggableWindow3.style.top = (e.clientY - offsetY3) + 'px';
    }
});




canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);





function startDrawing(e) {
    drawing = true;
    draw(e); // To allow a single click to paint a pixel
  }
  
  function stopDrawing() {
    drawing = false;
    ctx.beginPath(); // Ensures the next line drawn is independent from the last
  }
  
  function draw(e) {
    if (!drawing) return;
  
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
  
    // Calculate rounded coordinates
    const roundedX = Math.floor(x);
    const roundedY = Math.floor(y);
  
    // Draw at the exact position with full opacity
    drawPixel(roundedX, roundedY, 1);
  
    // Draw at top, left, right, and bottom positions with 50% opacity
    drawPixel(roundedX, roundedY - 1, 0.5); // Top
    drawPixel(roundedX, roundedY + 1, 0.5); // Bottom
    drawPixel(roundedX - 1, roundedY, 0.5); // Left
    drawPixel(roundedX + 1, roundedY, 0.5); // Right
  }
  
  
  // Helper function to draw a black rectangle (pixel)
  function drawPixel(x, y, alpha = 1) {
    if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
      ctx.globalAlpha = alpha; // Set the alpha (opacity)
      ctx.fillStyle = 'black';
      ctx.fillRect(x, y, 1, 1);
      ctx.globalAlpha = 1; // Reset alpha to default
    }
  }

// const logButton = document.getElementById('logButton');
// logButton.addEventListener('click', logCanvasData);
// function logCanvasData() {
//     const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
//     let grid = [];
  
//     for (let y = 0; y < canvas.height; y++) {
//       let row = [];
      
//       for (let x = 0; x < canvas.width; x++) {
//         // Calculate the index into the pixelData array
//         const idx = (y * canvas.width + x) * 4;
        
//         // Get the alpha component of the pixel
//         const alpha = pixelData[idx + 3];
        
//         // Add the alpha value to the row
//         row.push(alpha);
//       }
      
//       // Add the row to the grid
//       grid.push(row);
//     }
    
//     // Log the grid to the console
//     console.log(grid);
//   }
  
  
  
  