const canvas = document.getElementById('pixelCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;
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

const logButton = document.getElementById('logButton');
logButton.addEventListener('click', logCanvasData);
function logCanvasData() {
    const pixelData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let grid = [];
  
    for (let y = 0; y < canvas.height; y++) {
      let row = [];
      
      for (let x = 0; x < canvas.width; x++) {
        // Calculate the index into the pixelData array
        const idx = (y * canvas.width + x) * 4;
        
        // Get the alpha component of the pixel
        const alpha = pixelData[idx + 3];
        
        // Add the alpha value to the row
        row.push(alpha);
      }
      
      // Add the row to the grid
      grid.push(row);
    }
    
    // Log the grid to the console
    console.log(grid);
  }
  
  
  
  