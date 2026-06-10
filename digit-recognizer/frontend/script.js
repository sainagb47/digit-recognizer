const API_BASE = "https://digit-recognizer-api-0gft.onrender.com";
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const viewportCanvas = document.getElementById("viewport-canvas");
const viewportCtx = viewportCanvas.getContext("2d");
const brushPreview = document.getElementById("brush-preview");
const brushSizeSlider = document.getElementById("brush-size");
const brushSizeVal = document.getElementById("brush-size-val");
const connectionStatus = document.getElementById("connection-status");
const connectionText = connectionStatus.querySelector(".status-text");
const predictBtn = document.getElementById("predict-btn");
const probabilityBarsContainer = document.getElementById("probability-bars");

// Drawing state
let drawing = false;
let tool = "draw"; // "draw" or "erase"
let brushSize = 16;
let autoPredict = true;
let lastX = 0;
let lastY = 0;

// Undo/Redo stacks
const maxStackSize = 25;
let undoStack = [];
let redoStack = [];

// Initialize Canvas
function initCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    saveState(); // Save initial blank state
    updateViewport();
}

// Generate probability rows in UI
function initProbabilityBars() {
    probabilityBarsContainer.innerHTML = "";
    for (let i = 0; i < 10; i++) {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.id = `prob-row-${i}`;
        
        row.innerHTML = `
            <span class="prob-digit">${i}</span>
            <div class="prob-bar-wrapper">
                <div class="prob-bar-fill" id="prob-bar-${i}"></div>
            </div>
            <span class="prob-val" id="prob-val-${i}">0%</span>
        `;
        probabilityBarsContainer.appendChild(row);
    }
}

// Save current canvas state to undo stack
function saveState() {
    if (undoStack.length >= maxStackSize) {
        undoStack.shift();
    }
    undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    redoStack = []; // Clear redo stack on new action
}

// Canvas event listeners
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDrawing);

// Touch support for tablets/mobile devices
canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    startDrawing({
        offsetX: touch.clientX - rect.left,
        offsetY: touch.clientY - rect.top
    });
});
canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    draw({
        offsetX: touch.clientX - rect.left,
        offsetY: touch.clientY - rect.top
    });
});
canvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    stopDrawing();
});

// Brush indicator preview listeners
canvas.addEventListener("mouseenter", () => brushPreview.style.display = "block");
canvas.addEventListener("mouseleave", () => brushPreview.style.display = "none");
canvas.addEventListener("mousemove", updateBrushPreviewPosition);

function updateBrushPreviewPosition(e) {
    brushPreview.style.left = `${e.offsetX + canvas.offsetLeft}px`;
    brushPreview.style.top = `${e.offsetY + canvas.offsetTop}px`;
}

function startDrawing(e) {
    drawing = true;
    saveState();
    [lastX, lastY] = [e.offsetX, e.offsetY];
    
    // Draw a point immediately on click
    ctx.fillStyle = tool === "draw" ? "white" : "black";
    ctx.beginPath();
    ctx.arc(lastX, lastY, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    updateViewport();
}

function draw(e) {
    if (!drawing) return;
    
    ctx.strokeStyle = tool === "draw" ? "white" : "black";
    ctx.fillStyle = tool === "draw" ? "white" : "black";
    ctx.lineWidth = brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
    updateViewport();
    
    if (autoPredict) {
        throttlePredict();
    }
}

function stopDrawing() {
    if (!drawing) return;
    drawing = false;
    updateViewport();
    if (autoPredict) {
        predict();
    }
}

// Downsample drawing canvas to 28x28 viewport
function updateViewport() {
    // Fill background black, then draw downsampled version of main canvas
    viewportCtx.fillStyle = "black";
    viewportCtx.fillRect(0, 0, 28, 28);
    viewportCtx.drawImage(canvas, 0, 0, 28, 28);
}

// Clear canvas
function clearCanvas() {
    saveState();
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updateViewport();
    resetPredictions();
}

// Tool setting (Draw vs Erase)
function setTool(t) {
    tool = t;
    document.getElementById("tool-draw").classList.toggle("active", tool === "draw");
    document.getElementById("tool-erase").classList.toggle("active", tool === "erase");
    
    if (tool === "erase") {
        brushPreview.style.borderColor = "rgba(239, 68, 68, 0.8)";
    } else {
        brushPreview.style.borderColor = "rgba(255, 255, 255, 0.8)";
    }
}

// Brush size adjustments
function updateBrushSize(size) {
    brushSize = parseInt(size);
    brushSizeVal.innerText = `${size}px`;
    brushPreview.style.width = `${size}px`;
    brushPreview.style.height = `${size}px`;
}

// Toggle auto predictions
function toggleAutoPredict(checked) {
    autoPredict = checked;
    predictBtn.classList.toggle("hidden", autoPredict);
}

// Undo action
function undo() {
    if (undoStack.length > 0) {
        redoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
        const state = undoStack.pop();
        ctx.putImageData(state, 0, 0);
        updateViewport();
        if (autoPredict) predict();
    }
}

// Redo action
function redo() {
    if (redoStack.length > 0) {
        undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
        const state = redoStack.pop();
        ctx.putImageData(state, 0, 0);
        updateViewport();
        if (autoPredict) predict();
    }
}

// Keyboard shortcuts (Ctrl+Z, Ctrl+Y)
window.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "z") {
        e.preventDefault();
        undo();
    } else if (e.ctrlKey && e.key === "y") {
        e.preventDefault();
        redo();
    }
});

// Predictions API interface
let predictTimeout = null;
function throttlePredict() {
    if (predictTimeout) return;
    predictTimeout = setTimeout(() => {
        predict();
        predictTimeout = null;
    }, 150); // call at most once every 150ms during drawing
}

function predictManual() {
    predict();
}

function predict() {
    // If the canvas is completely black (unpainted), avoid hitting the server and reset guesses
    if (isCanvasBlank()) {
        resetPredictions();
        return;
    }

    const image = canvas.toDataURL("image/png");
    
    fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: image })
    })
    .then(res => {
        if (!res.ok) throw new Error("Server error");
        return res.json();
    })
    .then(data => {
        updateConnectionStatus(true);
        displayResults(data);
    })
    .catch(err => {
        updateConnectionStatus(false);
        console.error("Prediction failed:", err);
    });
}

function isCanvasBlank() {
    const buffer = new Uint32Array(ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer);
    // Check if any pixels are not black (black is 0xFF000000 in little endian or simply 0, we can check for values != 0)
    return !buffer.some(color => (color & 0x00FFFFFF) !== 0);
}

function displayResults(data) {
    const topGuess = data.first_guess;
    const secondGuess = data.second_guess;
    const probs = data.probabilities;
    
    // Update main display
    document.getElementById("first-guess").innerText = topGuess;
    document.getElementById("second-guess").innerText = secondGuess;
    
    const topConfidence = (probs[topGuess] * 100).toFixed(1);
    document.getElementById("confidence-val").innerText = `${topConfidence}%`;
    
    // Update probability distribution chart
    probs.forEach((p, i) => {
        const barFill = document.getElementById(`prob-bar-${i}`);
        const valLabel = document.getElementById(`prob-val-${i}`);
        const row = document.getElementById(`prob-row-${i}`);
        
        const pct = (p * 100).toFixed(1);
        barFill.style.width = `${pct}%`;
        valLabel.innerText = `${pct}%`;
        
        // Highlight top prediction row
        if (i === topGuess) {
            row.classList.add("top-probability");
        } else {
            row.classList.remove("top-probability");
        }
    });
}

function resetPredictions() {
    document.getElementById("first-guess").innerText = "-";
    document.getElementById("second-guess").innerText = "-";
    document.getElementById("confidence-val").innerText = "0.0%";
    
    for (let i = 0; i < 10; i++) {
        document.getElementById(`prob-bar-${i}`).style.width = "0%";
        document.getElementById(`prob-val-${i}`).innerText = "0%";
        document.getElementById(`prob-row-${i}`).classList.remove("top-probability");
    }
}

// Connection Health Checking
function updateConnectionStatus(isConnected) {
    if (isConnected) {
        connectionStatus.className = "status-badge online";
        connectionText.innerText = "Server Connected";
    } else {
        connectionStatus.className = "status-badge offline";
        connectionText.innerText = "Server Disconnected";
    }
}

function checkServerHealth() {
    fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAFUlEQVR42mNkWPifwEgoGCkqHGmOADcECV5a5zoVAAAAAElFTkSuQmCC" })
    })
    .then(res => {
        updateConnectionStatus(res.ok);
    })
    .catch(() => {
        updateConnectionStatus(false);
    });
}

// Setup & Initialization
window.addEventListener("DOMContentLoaded", () => {
    initCanvas();
    initProbabilityBars();
    updateBrushSize(brushSizeSlider.value);
    
    // Periodically check server connection
    checkServerHealth();
    setInterval(checkServerHealth, 5000);
});
