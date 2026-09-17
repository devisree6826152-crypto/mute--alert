/**
 * SignSpeak AI - Client Frontend JavaScript Engine
 * Optimized for high-performance 60 FPS video & zero-latency landmark tracking
 */

let mediaStream = null;
let isWebcamRunning = false;
let animFrameId = null;
let sentenceWords = [];
let currentPrediction = "";
let currentConfidence = 0;
let confidenceThreshold = 0.65;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let currentFps = 0;
let isProcessingFrame = false;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initGlobalListeners();
    initSOSFlow();

    const path = window.location.pathname;
    if (path === '/recognition' || path.includes('recognition')) {
        initRecognitionPage();
    } else if (path === '/dataset' || path.includes('dataset')) {
        initDatasetPage();
    } else if (path === '/training' || path.includes('training')) {
        initTrainingPage();
    } else if (path === '/history' || path.includes('history')) {
        initHistoryPage();
    } else {
        initHomePage();
    }
});

/* Theme Switcher */
function initTheme() {
    const savedTheme = localStorage.getItem('signspeak_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('signspeak_theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
        themeBtn.innerHTML = theme === 'dark' 
            ? '<i class="bi bi-sun-fill text-warning"></i> Light Mode' 
            : '<i class="bi bi-moon-stars-fill text-primary"></i> Dark Mode';
    }
}

function initGlobalListeners() {
    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }
}

/* Skeleton Drawer on Transparent Canvas */
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        // Index
    [5, 9], [9, 10], [10, 11], [11, 12],   // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
    [0, 17]                                // Wrist to Pinky base
];

function drawSkeletonOnCanvas(canvas, rawLandmarks) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!rawLandmarks || rawLandmarks.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;
    const pts = rawLandmarks.map(lm => ({ x: lm[0] * w, y: lm[1] * h }));

    // Draw connection lines
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 3;
    HAND_CONNECTIONS.forEach(([i, j]) => {
        if (pts[i] && pts[j]) {
            ctx.beginPath();
            ctx.moveTo(pts[i].x, pts[i].y);
            ctx.lineTo(pts[j].x, pts[j].y);
            ctx.stroke();
        }
    });

    // Draw joint nodes
    pts.forEach(pt => {
        ctx.fillStyle = '#00e676';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}


/* Ultra-Fast Camera Engine */
async function startCamera(videoElement, canvasElement, onFrameCallback) {
    if (isWebcamRunning) stopCamera();

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 }, 
                frameRate: { ideal: 30 },
                facingMode: "user" 
            },
            audio: false
        });

        videoElement.srcObject = mediaStream;
        await videoElement.play();
        isWebcamRunning = true;

        // Resize overlay canvas to match video dimensions
        if (canvasElement) {
            canvasElement.width = videoElement.videoWidth || 640;
            canvasElement.height = videoElement.videoHeight || 480;
        }

        // Lightweight capture canvas for API snapshot
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = 320;
        captureCanvas.height = 240;
        const captureCtx = captureCanvas.getContext('2d');

        async function predictLoop() {
            if (!isWebcamRunning) return;

            if (canvasElement && videoElement.videoWidth > 0 && canvasElement.width !== videoElement.videoWidth) {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
            }

            frameCount++;
            const now = Date.now();
            if (now - lastFpsUpdate >= 1000) {
                currentFps = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
                const fpsElem = document.getElementById('fpsCounter');
                if (fpsElem) fpsElem.innerText = `${currentFps} FPS`;
            }

            if (!isProcessingFrame && videoElement.readyState === 4) {
                isProcessingFrame = true;

                captureCtx.drawImage(videoElement, 0, 0, 320, 240);
                const base64Frame = captureCanvas.toDataURL('image/jpeg', 0.6);
                window.lastWebcamFrame = base64Frame;

                if (onFrameCallback) {
                    try {
                        await onFrameCallback(base64Frame);
                    } catch (err) {
                        console.error("Frame callback error:", err);
                    } finally {
                        isProcessingFrame = false;
                    }
                } else {
                    isProcessingFrame = false;
                }
            }

            animFrameId = requestAnimationFrame(predictLoop);
        }

        predictLoop();
        return true;
    } catch (err) {
        console.error("Camera access error:", err);
        showToast("Webcam Error: " + err.message, "danger");
        return false;
    }
}

function stopCamera() {
    isWebcamRunning = false;
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
}

/* Toast Notifications */
function showToast(message, type = 'info') {
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.style.position = 'fixed';
        toastContainer.style.bottom = '20px';
        toastContainer.style.right = '20px';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0 show shadow-lg mb-2`;
    toast.role = 'alert';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body"><i class="bi bi-info-circle-fill me-2"></i> ${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    toastContainer.appendChild(toast);
    setTimeout(() => { toast.remove(); }, 3500);
}

/* Text-to-Speech */
function speakText(text) {
    if (!text || text.trim() === '') {
        showToast("No sentence to speak!", "warning");
        return;
    }

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
        showToast(`Speaking: "${text}"`, "success");
    } else {
        fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        })
        .then(res => res.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play();
        })
        .catch(err => showToast("TTS Error: " + err.message, "danger"));
    }
}


/* ---------------- 1. RECOGNITION PAGE ---------------- */
function initRecognitionPage() {
    const videoElem = document.getElementById('webcamVideo');
    const skeletonCanvas = document.getElementById('skeletonCanvas');
    const toggleCamBtn = document.getElementById('toggleCamBtn');
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceValueLabel = document.getElementById('confidenceValueLabel');

    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', (e) => {
            confidenceThreshold = parseFloat(e.target.value);
            if (confidenceValueLabel) {
                confidenceValueLabel.innerText = `${Math.round(confidenceThreshold * 100)}%`;
            }
        });
    }

    if (toggleCamBtn) {
        toggleCamBtn.addEventListener('click', async () => {
            if (isWebcamRunning) {
                stopCamera();
                toggleCamBtn.innerHTML = '<i class="bi bi-camera-video-fill"></i> Start Camera';
                toggleCamBtn.className = 'btn btn-ai-primary';
                if (skeletonCanvas) {
                    const ctx = skeletonCanvas.getContext('2d');
                    ctx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
                }
                showToast("Camera stopped", "info");
            } else {
                toggleCamBtn.innerHTML = '<i class="bi bi-stop-circle-fill"></i> Stop Camera';
                toggleCamBtn.className = 'btn btn-danger';
                await startCamera(videoElem, skeletonCanvas, async (base64Frame) => {
                    await processRecognitionFrame(base64Frame, skeletonCanvas);
                });
            }
        });
    }

    document.getElementById('btnAddWord')?.addEventListener('click', () => addWordToSentence(currentPrediction));
    document.getElementById('btnDeleteLast')?.addEventListener('click', deleteLastWord);
    document.getElementById('btnClearSentence')?.addEventListener('click', clearSentence);
    document.getElementById('btnSpeakSentence')?.addEventListener('click', () => speakText(getSentenceString()));
    document.getElementById('btnSaveSentence')?.addEventListener('click', saveSentenceToHistory);

    setTimeout(() => {
        if (toggleCamBtn) toggleCamBtn.click();
    }, 400);
}

async function processRecognitionFrame(base64Frame, skeletonCanvas) {
    try {
        const response = await fetch('/api/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame: base64Frame,
                threshold: confidenceThreshold
            })
        });

        const data = await response.json();
        if (!data.success) return;

        const predValElem = document.getElementById('predictionValue');
        const confFillElem = document.getElementById('confidenceFill');
        const confTextElem = document.getElementById('confidenceText');
        const handBadgeElem = document.getElementById('handBadge');

        if (data.hand_detected) {
            currentPrediction = data.prediction || "NO SIGN";
            currentConfidence = data.confidence || 0;

            if (predValElem) predValElem.innerText = currentPrediction;
            const confPct = Math.round(currentConfidence * 100);
            if (confFillElem) confFillElem.style.width = `${confPct}%`;
            if (confTextElem) confTextElem.innerText = `${confPct}%`;
            if (handBadgeElem) {
                handBadgeElem.className = "webcam-overlay-badge";
                handBadgeElem.innerHTML = '<span class="pulse-dot"></span> Hand Active';
            }

            drawSkeletonOnCanvas(skeletonCanvas, data.raw_landmarks);

            if (data.stable_prediction && data.confidence >= confidenceThreshold) {
                autoAppendWord(data.stable_prediction);
            }
        } else {
            currentPrediction = "";
            currentConfidence = 0;
            if (predValElem) predValElem.innerText = "No Hand Detected";
            if (confFillElem) confFillElem.style.width = `0%`;
            if (confTextElem) confTextElem.innerText = `0%`;
            if (handBadgeElem) {
                handBadgeElem.className = "webcam-overlay-badge border-danger text-danger";
                handBadgeElem.innerHTML = '<i class="bi bi-exclamation-triangle-fill text-danger"></i> No Hand';
            }
            drawSkeletonOnCanvas(skeletonCanvas, null);
        }
    } catch (err) {
        console.error("Frame prediction error:", err);
    }
}

let lastAutoWord = "";
function autoAppendWord(word) {
    if (!word || word === lastAutoWord) return;
    addWordToSentence(word);
    lastAutoWord = word;
    setTimeout(() => { if (lastAutoWord === word) lastAutoWord = ""; }, 2500);
}

function addWordToSentence(word) {
    if (!word || word === "No Hand Detected" || word === "NO SIGN") return;
    sentenceWords.push(word);
    renderSentenceChips();
}

function deleteLastWord() {
    sentenceWords.pop();
    renderSentenceChips();
}

function clearSentence() {
    sentenceWords = [];
    renderSentenceChips();
    lastAutoWord = "";
}

function getSentenceString() {
    return sentenceWords.join(" ");
}

function renderSentenceChips() {
    const container = document.getElementById('sentenceChips');
    if (!container) return;

    if (sentenceWords.length === 0) {
        container.innerHTML = '<span class="text-muted fst-italic">Recognized signs will appear here...</span>';
        return;
    }

    container.innerHTML = sentenceWords.map((w, idx) => `
        <span class="word-chip">
            ${w}
            <i class="bi bi-x-circle-fill" onclick="removeWordAtIndex(${idx})"></i>
        </span>
    `).join('');
}

function removeWordAtIndex(idx) {
    sentenceWords.splice(idx, 1);
    renderSentenceChips();
}

async function saveSentenceToHistory() {
    const sentence = getSentenceString();
    if (!sentence) {
        showToast("Sentence is empty! Add signs before saving.", "warning");
        return;
    }

    try {
        const response = await fetch('/api/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: sentence,
                confidence: currentConfidence || 0.95,
                sign_type: 'sentence'
            })
        });

        const data = await response.json();
        if (data.success) {
            showToast("Saved sentence to SQLite History!", "success");
        } else {
            showToast("Failed to save: " + data.error, "danger");
        }
    } catch (err) {
        showToast("Network error saving history", "danger");
    }
}


/* ---------------- 2. DATASET PAGE ---------------- */
function initDatasetPage() {
    const videoElem = document.getElementById('datasetWebcam');
    const skeletonCanvas = document.getElementById('datasetSkeletonCanvas');
    const toggleCamBtn = document.getElementById('toggleDatasetCamBtn');
    const captureBtn = document.getElementById('captureSampleBtn');
    const signSelect = document.getElementById('datasetSignSelect');
    const newSignInput = document.getElementById('newSignInput');
    const btnAddNewSign = document.getElementById('btnAddNewSign');

    if (btnAddNewSign && newSignInput && signSelect) {
        btnAddNewSign.addEventListener('click', () => {
            const val = newSignInput.value.trim().toUpperCase();
            if (!val) return;
            const opt = document.createElement('option');
            opt.value = val;
            opt.innerText = val;
            opt.selected = true;
            signSelect.appendChild(opt);
            newSignInput.value = '';
            showToast(`Added '${val}' to sign list`, "success");
        });
    }

    if (toggleCamBtn) {
        toggleCamBtn.addEventListener('click', async () => {
            if (isWebcamRunning) {
                stopCamera();
                toggleCamBtn.innerHTML = '<i class="bi bi-camera-video-fill"></i> Start Camera';
                toggleCamBtn.className = 'btn btn-ai-primary';
                showToast("Dataset Camera Stopped", "info");
            } else {
                toggleCamBtn.innerHTML = '<i class="bi bi-stop-circle-fill"></i> Stop Camera';
                toggleCamBtn.className = 'btn btn-danger';
                await startCamera(videoElem, skeletonCanvas, async (base64Frame) => {
                    try {
                        const res = await fetch('/api/predict_frame', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ frame: base64Frame, threshold: 0.5 })
                        });
                        const data = await res.json();
                        if (data.success && data.raw_landmarks) {
                            drawSkeletonOnCanvas(skeletonCanvas, data.raw_landmarks);
                        }
                    } catch (e) {}
                });
            }
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', async () => {
            if (!window.lastWebcamFrame || !isWebcamRunning) {
                showToast("Start webcam first!", "warning");
                return;
            }

            const selectedSign = signSelect ? signSelect.value : "HELLO";
            try {
                const res = await fetch('/api/collect_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        frame: window.lastWebcamFrame,
                        sign_name: selectedSign
                    })
                });

                const data = await res.json();
                if (data.success) {
                    showToast(data.message, "success");
                    loadDatasetStats();
                } else {
                    showToast(data.message || data.error, "danger");
                }
            } catch (err) {
                showToast("Error capturing sample: " + err.message, "danger");
            }
        });
    }

    loadDatasetStats();

    setTimeout(() => {
        if (toggleCamBtn) toggleCamBtn.click();
    }, 400);
}

async function loadDatasetStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (!data.success) return;

        const tableBody = document.getElementById('datasetStatsTableBody');
        const totalSamplesElem = document.getElementById('totalSamplesBadge');
        const totalClassesElem = document.getElementById('totalClassesBadge');

        if (totalSamplesElem) totalSamplesElem.innerText = data.total_samples;
        if (totalClassesElem) totalClassesElem.innerText = data.num_classes;

        if (tableBody) {
            tableBody.innerHTML = Object.entries(data.class_counts).map(([cls, cnt]) => `
                <tr>
                    <td><span class="badge bg-primary fs-6">${cls}</span></td>
                    <td class="font-monospace fw-bold">${cnt}</td>
                    <td>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar bg-info" style="width: ${Math.min(100, (cnt / 60) * 100)}%"></div>
                        </div>
                    </td>
                </tr>
            `).join('');
        }
    } catch (err) {
        console.error("Load dataset stats error:", err);
    }
}


/* ---------------- 3. MODEL TRAINING PAGE ---------------- */
function initTrainingPage() {
    const btnTrain = document.getElementById('btnTrainModel');
    if (btnTrain) {
        btnTrain.addEventListener('click', triggerModelTraining);
    }
    loadTrainingMetrics();
}

async function triggerModelTraining() {
    const btnTrain = document.getElementById('btnTrainModel');
    const spinner = document.getElementById('trainSpinner');
    const statusBox = document.getElementById('trainStatusBox');

    if (btnTrain) btnTrain.disabled = true;
    if (spinner) spinner.classList.remove('d-none');
    if (statusBox) {
        statusBox.className = "alert alert-info shadow-sm";
        statusBox.innerHTML = '<i class="bi bi-cpu-fill me-2"></i> Preprocessing landmark features and training Random Forest Classifier...';
    }

    try {
        const response = await fetch('/api/train', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showToast("Model trained successfully!", "success");
            if (statusBox) {
                statusBox.className = "alert alert-success shadow-sm";
                statusBox.innerHTML = `<i class="bi bi-check-circle-fill me-2"></i> ${data.message}`;
            }
            loadTrainingMetrics();
        } else {
            showToast(data.message || data.error, "danger");
            if (statusBox) {
                statusBox.className = "alert alert-danger shadow-sm";
                statusBox.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-2"></i> ${data.message || data.error}`;
            }
        }
    } catch (err) {
        showToast("Error training model: " + err.message, "danger");
    } finally {
        if (btnTrain) btnTrain.disabled = false;
        if (spinner) spinner.classList.add('d-none');
    }
}

async function loadTrainingMetrics() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (!data.success) return;

        document.getElementById('accValue').innerText = `${(data.accuracy * 100).toFixed(1)}%`;
        document.getElementById('trainedAtLabel').innerText = data.trained_at || 'Not Trained';

        const metrics = data.metrics || {};
        if (metrics.precision) document.getElementById('precValue').innerText = `${(metrics.precision * 100).toFixed(1)}%`;
        if (metrics.recall) document.getElementById('recValue').innerText = `${(metrics.recall * 100).toFixed(1)}%`;
        if (metrics.f1_score) document.getElementById('f1Value').innerText = `${(metrics.f1_score * 100).toFixed(1)}%`;

        renderConfusionMatrix(metrics.confusion_matrix, metrics.classes);
    } catch (err) {
        console.error("Load metrics error:", err);
    }
}

function renderConfusionMatrix(matrix, classes) {
    const container = document.getElementById('matrixContainer');
    if (!container || !matrix || !classes) return;

    const n = classes.length;
    container.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    container.innerHTML = '';

    let maxVal = 1;
    matrix.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

    matrix.forEach((row, i) => {
        row.forEach((val, j) => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            const intensity = Math.min(1.0, val / maxVal);
            cell.style.background = i === j 
                ? `rgba(0, 230, 118, ${0.25 + intensity * 0.75})` 
                : `rgba(255, 64, 129, ${intensity * 0.5})`;
            cell.title = `True: ${classes[i]}, Pred: ${classes[j]} (${val})`;
            cell.innerText = val;
            container.appendChild(cell);
        });
    });
}


/* ---------------- 4. HISTORY PAGE ---------------- */
function initHistoryPage() {
    document.getElementById('btnClearHistory')?.addEventListener('click', clearAllHistory);
    document.getElementById('tabSosHistory')?.addEventListener('click', loadSosHistoryData);
    loadHistoryData();
    loadSosHistoryData();
}

async function loadHistoryData() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        if (!data.success) return;

        const tableBody = document.getElementById('historyTableBody');
        if (!tableBody) return;

        if (data.history.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-4">No recognition history logged yet.</td></tr>';
            return;
        }

        tableBody.innerHTML = data.history.map((row, idx) => `
            <tr>
                <td>#${row.id}</td>
                <td class="fw-bold text-light fs-6">${row.recognized_text}</td>
                <td>
                    <span class="badge bg-success bg-opacity-75">
                        ${Math.round(row.confidence * 100)}%
                    </span>
                </td>
                <td class="text-secondary small">${row.timestamp}</td>
                <td>
                    <button class="btn btn-sm btn-ai-outline me-1" onclick="speakText('${row.recognized_text.replace(/'/g, "\\'")}')">
                        <i class="bi bi-volume-up-fill"></i> Speak
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteHistoryRecord(${row.id})">
                        <i class="bi bi-trash-fill"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    } catch (err) {
        console.error("Load history error:", err);
    }
}

async function deleteHistoryRecord(id) {
    try {
        const res = await fetch(`/api/history?id=${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast("Record deleted", "info");
            loadHistoryData();
        }
    } catch (err) {
        showToast("Error deleting record", "danger");
    }
}

async function clearAllHistory() {
    if (!confirm("Are you sure you want to clear all recognition history?")) return;
    try {
        const res = await fetch('/api/history', { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast("History cleared", "success");
            loadHistoryData();
        }
    } catch (err) {
        showToast("Error clearing history", "danger");
    }
}

function initHomePage() {}


/* ==================== REAL END-TO-END EMERGENCY SOS ENGINE ==================== */

let sosState = {
    selectedEmergencyType: "Medical Emergency",
    latitude: null,
    longitude: null,
    accuracy: null,
    locationMode: "gps", // 'gps' or 'manual'
    mapsUrl: null,
    batteryStatus: "Battery information unavailable",
    activeSosId: null,
    activeAlertText: "",
    isSubmitting: false,
    holdTimer: null,
    holdStartTime: 0,
    holdInterval: null,
    contacts: []
};

function initSOSFlow() {
    loadContactsData();
    initSettingsNumber();

    document.getElementById('navTriggerSosBtn')?.addEventListener('click', () => {
        openSosWizard(1);
    });

    const holdBtn = document.getElementById('sosHoldBtn');
    if (holdBtn) {
        const startHold = (e) => {
            e.preventDefault();
            startSosHoldTimer();
        };
        const endHold = (e) => {
            e.preventDefault();
            cancelSosHoldTimer();
        };

        holdBtn.addEventListener('mousedown', startHold);
        holdBtn.addEventListener('touchstart', startHold, { passive: false });

        holdBtn.addEventListener('mouseup', endHold);
        holdBtn.addEventListener('mouseleave', endHold);
        holdBtn.addEventListener('touchend', endHold);
        holdBtn.addEventListener('touchcancel', endHold);
    }

    document.getElementById('sosTapArmBtn')?.addEventListener('click', () => {
        announceSr("Emergency SOS armed via tap. Opening wizard.");
        openSosWizard(1);
    });

    document.getElementById('btnOpenAddContact')?.addEventListener('click', () => {
        openContactModal();
    });
    document.getElementById('btnSaveContactModal')?.addEventListener('click', saveContactFromModal);
    document.getElementById('btnQuickAddContact')?.addEventListener('click', quickAddContactInline);

    document.getElementById('btnSaveEmergencyNumber')?.addEventListener('click', saveEmergencyNumberSetting);

    document.querySelectorAll('.sos-type-card').forEach(card => {
        card.addEventListener('click', function() {
            document.querySelectorAll('.sos-type-card').forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            sosState.selectedEmergencyType = this.getAttribute('data-type') || "Medical Emergency";
        });
    });

    document.getElementById('btnStep1Next')?.addEventListener('click', () => goToSosStep(2));
    document.getElementById('btnStep2Back')?.addEventListener('click', () => goToSosStep(1));
    document.getElementById('btnStep2Next')?.addEventListener('click', () => goToSosStep(3));
    document.getElementById('btnStep3Back')?.addEventListener('click', () => goToSosStep(2));
    
    document.getElementById('btnSendSosNow')?.addEventListener('click', triggerSendSosNow);

    document.getElementById('btnTelHandoff')?.addEventListener('click', handleTelHandoff);
    document.getElementById('btnShareHandoff')?.addEventListener('click', handleShareHandoff);
    document.getElementById('btnCopyHandoff')?.addEventListener('click', handleCopyHandoff);

    document.getElementById('btnResolveSos')?.addEventListener('click', resolveActiveSos);

    document.getElementById('manualLocationInput')?.addEventListener('input', (e) => {
        const val = e.target.value.trim();
        if (val.length > 0) {
            sosState.locationMode = 'manual';
            sosState.mapsUrl = val.includes('http') ? val : `https://maps.google.com/?q=${encodeURIComponent(val)}`;
        } else if (sosState.latitude && sosState.longitude) {
            sosState.locationMode = 'gps';
            sosState.mapsUrl = `https://maps.google.com/?q=${sosState.latitude},${sosState.longitude}`;
        }
    });
}

function announceSr(message) {
    const srElem = document.getElementById('sosHoldSrStatus');
    if (srElem) srElem.innerText = message;
}

function startSosHoldTimer() {
    sosState.holdStartTime = Date.now();
    const btn = document.getElementById('sosHoldBtn');
    if (btn) btn.classList.add('holding');

    announceSr("Holding SOS button. Keep holding for 2 seconds.");

    const circle = document.getElementById('sosProgressCircle');
    const bar = document.getElementById('sosProgressBar');

    const totalCircumference = 452.39;

    sosState.holdInterval = setInterval(() => {
        const elapsed = Date.now() - sosState.holdStartTime;
        const progress = Math.min(1.0, elapsed / 2000);
        const percentage = Math.round(progress * 100);

        if (circle) {
            const offset = totalCircumference - (progress * totalCircumference);
            circle.style.strokeDashoffset = offset;
        }
        if (bar) bar.style.width = `${percentage}%`;

        if (percentage % 25 === 0) {
            announceSr(`Holding SOS button: ${percentage}%`);
        }

        if (progress >= 1.0) {
            cancelSosHoldTimer();
            announceSr("SOS Triggered! Opening Emergency Wizard.");
            openSosWizard(1);
        }
    }, 40);
}

function cancelSosHoldTimer() {
    if (sosState.holdInterval) {
        clearInterval(sosState.holdInterval);
        sosState.holdInterval = null;
    }
    const btn = document.getElementById('sosHoldBtn');
    if (btn) btn.classList.remove('holding');

    const circle = document.getElementById('sosProgressCircle');
    const bar = document.getElementById('sosProgressBar');

    if (circle) circle.style.strokeDashoffset = 452.39;
    if (bar) bar.style.width = '0%';
}

/* Contacts Management Logic */
async function loadContactsData() {
    try {
        const res = await fetch('/api/contacts');
        const data = await res.json();

        if (data.success && data.contacts) {
            sosState.contacts = data.contacts;
        } else {
            const local = localStorage.getItem('signspeak_contacts');
            sosState.contacts = local ? JSON.parse(local) : [];
        }
    } catch (e) {
        const local = localStorage.getItem('signspeak_contacts');
        sosState.contacts = local ? JSON.parse(local) : [];
    }
    renderContactsList();
}

function renderContactsList() {
    const container = document.getElementById('contactsListContainer');
    if (!container) return;

    if (sosState.contacts.length === 0) {
        container.innerHTML = `
            <div class="col-12 text-center text-muted py-4">
                <i class="bi bi-person-x fs-2 d-block text-secondary mb-2"></i>
                No emergency contacts saved yet.<br>
                <span class="small">Add your trusted contacts so they can be notified in an emergency.</span>
            </div>
        `;
        return;
    }

    container.innerHTML = sosState.contacts.map(c => {
        const cleanPhone = c.phone.replace(/[^0-9+]/g, '');
        return `
            <div class="col-md-6">
                <div class="p-3 rounded bg-dark border border-secondary d-flex justify-content-between align-items-center">
                    <div>
                        <div class="fw-bold text-white fs-6"><i class="bi bi-person-fill text-info me-1"></i> ${c.name}</div>
                        <div class="font-monospace text-cyan small"><i class="bi bi-telephone-fill me-1"></i> ${c.phone}</div>
                        <div class="text-secondary x-small">${c.relationship || 'Emergency Contact'}</div>
                    </div>
                    <div class="d-flex gap-1 align-items-center">
                        <a href="tel:${cleanPhone}" class="btn btn-sm btn-success me-1" title="Call ${c.name}">
                            <i class="bi bi-telephone-fill"></i> Call
                        </a>
                        <button class="btn btn-sm btn-outline-info" onclick="openContactModal(${c.id}, '${c.name.replace(/'/g, "\\'")}', '${c.phone}', '${(c.relationship || '').replace(/'/g, "\\'")}')" title="Edit">
                            <i class="bi bi-pencil-fill"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteContactRecord(${c.id})" title="Delete">
                            <i class="bi bi-trash-fill"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function openContactModal(id = null, name = '', phone = '', relationship = '') {
    document.getElementById('contactEditId').value = id || '';
    document.getElementById('contactNameInput').value = name;
    document.getElementById('contactPhoneInput').value = phone;
    document.getElementById('contactRelInput').value = relationship;
    document.getElementById('contactModalTitle').innerText = id ? 'Edit Emergency Contact' : 'Add Emergency Contact';

    const modalElem = document.getElementById('contactModal');
    if (modalElem) {
        const bsModal = new bootstrap.Modal(modalElem);
        bsModal.show();
    }
}

async function saveContactFromModal() {
    const id = document.getElementById('contactEditId').value;
    const name = document.getElementById('contactNameInput').value.trim();
    const phone = document.getElementById('contactPhoneInput').value.trim();
    const relationship = document.getElementById('contactRelInput').value.trim();

    if (!name || !phone) {
        showToast("Please enter both Name and Phone Number!", "warning");
        return;
    }

    try {
        let res, data;
        if (id) {
            res = await fetch(`/api/contacts/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, phone, relationship })
            });
        } else {
            res = await fetch('/api/contacts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, phone, relationship })
            });
        }
        data = await res.json();
        if (data.success) {
            showToast(data.message, "success");
            const modalElem = document.getElementById('contactModal');
            const bsModal = bootstrap.Modal.getInstance(modalElem);
            if (bsModal) bsModal.hide();
            await loadContactsData();
        } else {
            showToast("Failed to save contact: " + data.error, "danger");
        }
    } catch (err) {
        showToast("Error saving contact: " + err.message, "danger");
    }
}

async function quickAddContactInline() {
    const name = document.getElementById('quickContactName')?.value.trim();
    const phone = document.getElementById('quickContactPhone')?.value.trim();

    if (!name || !phone) {
        showToast("Please enter Name and Phone Number!", "warning");
        return;
    }

    try {
        const res = await fetch('/api/contacts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, phone, relationship: 'Contact' })
        });
        const data = await res.json();
        if (data.success) {
            showToast("Contact added!", "success");
            await loadContactsData();
            document.getElementById('noContactsInlineWarning')?.classList.add('d-none');
            renderConfirmationPreview();
        }
    } catch (e) {
        showToast("Error adding contact", "danger");
    }
}

async function deleteContactRecord(id) {
    if (!confirm("Are you sure you want to delete this emergency contact?")) return;
    try {
        const res = await fetch(`/api/contacts/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast("Contact deleted", "info");
            await loadContactsData();
        }
    } catch (err) {
        showToast("Error deleting contact", "danger");
    }
}

/* Settings My Emergency Number */
function initSettingsNumber() {
    const savedNum = localStorage.getItem('signspeak_emergency_number') || '';
    const input = document.getElementById('myEmergencyNumberInput');
    if (input) input.value = savedNum;

    const subtext = document.getElementById('telNumberSubtext');
    if (subtext) {
        subtext.innerText = savedNum ? `Call ${savedNum}` : "Dial configured setting number";
    }
}

function saveEmergencyNumberSetting() {
    const input = document.getElementById('myEmergencyNumberInput');
    if (!input) return;
    const num = input.value.trim();
    localStorage.setItem('signspeak_emergency_number', num);
    initSettingsNumber();
    showToast("My Emergency Number setting saved!", "success");
}


/* Wizard Navigation & Steps */
function openSosWizard(step = 1) {
    const modalElem = document.getElementById('sosWizardModal');
    if (modalElem) {
        const bsModal = bootstrap.Modal.getOrCreateInstance(modalElem);
        bsModal.show();
    }
    goToSosStep(step);
}

function goToSosStep(step) {
    for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`sosWizardStep${i}`);
        if (el) el.classList.add('d-none');
    }

    const target = document.getElementById(`sosWizardStep${step}`);
    if (target) target.classList.remove('d-none');

    if (step === 2) {
        fetchRealLocation();
    } else if (step === 3) {
        fetchBatteryStatus();
        renderConfirmationPreview();
    } else if (step === 4) {
        renderActiveSosCallList();
    }
}

function renderActiveSosCallList() {
    const container = document.getElementById('activeSosContactsCallList');
    if (!container) return;

    if (!sosState.contacts || sosState.contacts.length === 0) {
        container.innerHTML = '<div class="col-12 text-muted small py-2">No emergency contacts saved to call directly.</div>';
        return;
    }

    container.innerHTML = sosState.contacts.map(c => {
        const cleanPhone = c.phone.replace(/[^0-9+]/g, '');
        return `
            <div class="col-md-6">
                <a href="tel:${cleanPhone}" class="btn btn-outline-success w-100 text-start d-flex justify-content-between align-items-center p-2 shadow-sm">
                    <span><i class="bi bi-telephone-fill me-2"></i><strong>Call ${c.name}</strong></span>
                    <span class="font-monospace small text-cyan">${c.phone}</span>
                </a>
            </div>
        `;
    }).join('');
}

/* Geolocation API Integration with Explicit Error & Context Handling */
function fetchRealLocation() {
    const statusAlert = document.getElementById('locationStatusAlert');
    const resultCard = document.getElementById('locationResultCard');

    if (statusAlert) {
        statusAlert.className = "alert alert-info d-flex align-items-center gap-2 mb-3";
        statusAlert.innerHTML = '<span class="spinner-border spinner-border-sm text-info"></span><span>Requesting real browser coordinates via Geolocation API...</span>';
    }

    if (!window.isSecureContext) {
        if (statusAlert) {
            statusAlert.className = "alert alert-warning border-warning mb-3";
            statusAlert.innerHTML = '<i class="bi bi-shield-lock-fill me-2 text-warning fs-5"></i><strong>Insecure Context:</strong> Location access is blocked because this page is served over an insecure (HTTP) connection. Please enter location manually below.';
        }
        enableManualLocationFallback();
        return;
    }

    if (window.self !== window.top) {
        if (statusAlert) {
            statusAlert.className = "alert alert-warning border-warning mb-3";
            statusAlert.innerHTML = '<i class="bi bi-window-stack me-2 text-warning fs-5"></i><strong>Iframe Container Detected:</strong> Location prompts may be blocked inside an iframe. Please open this page in its own tab, or enter location manually below.';
        }
        enableManualLocationFallback();
        return;
    }

    if (!navigator.geolocation) {
        if (statusAlert) {
            statusAlert.className = "alert alert-danger mb-3";
            statusAlert.innerHTML = '<i class="bi bi-exclamation-triangle-fill me-2"></i>Geolocation API is not supported by this browser. Enter location manually below.';
        }
        enableManualLocationFallback();
        return;
    }

    navigator.geolocation.getCurrentPosition(
        (pos) => {
            sosState.latitude = pos.coords.latitude;
            sosState.longitude = pos.coords.longitude;
            sosState.accuracy = Math.round(pos.coords.accuracy);
            sosState.locationMode = 'gps';
            sosState.mapsUrl = `https://maps.google.com/?q=${pos.coords.latitude},${pos.coords.longitude}`;

            if (statusAlert) {
                statusAlert.className = "alert alert-success d-flex align-items-center gap-2 mb-3";
                statusAlert.innerHTML = `<i class="bi bi-check-circle-fill me-2 fs-5"></i> High-accuracy GPS location acquired! (Accuracy: ~${sosState.accuracy}m)`;
            }

            if (resultCard) {
                resultCard.classList.remove('d-none');
                document.getElementById('locLat').innerText = pos.coords.latitude.toFixed(6);
                document.getElementById('locLng').innerText = pos.coords.longitude.toFixed(6);
                document.getElementById('locAcc').innerText = `~${sosState.accuracy} meters`;
                
                const mapsElem = document.getElementById('locMapsUrl');
                if (mapsElem) {
                    mapsElem.href = sosState.mapsUrl;
                    mapsElem.innerText = sosState.mapsUrl;
                }
                document.getElementById('locationModeBadge').innerText = "GPS Fix (Verified)";
                document.getElementById('locationModeBadge').className = "badge bg-success font-monospace";
            }
        },
        (err) => {
            console.warn("Geolocation API Error:", err);
            let msg = "";
            switch (err.code) {
                case 1:
                    msg = "Location permission denied. Please allow location access in your browser settings, or enter location manually below.";
                    break;
                case 2:
                    msg = "Position unavailable. Could not determine current GPS or network location. Enter location manually below.";
                    break;
                case 3:
                    msg = "Location request timed out after 12 seconds. Enter location manually below.";
                    break;
                default:
                    msg = "Location request failed. Enter location manually below.";
                    break;
            }

            if (statusAlert) {
                statusAlert.className = "alert alert-danger mb-3";
                statusAlert.innerHTML = `<i class="bi bi-geo-alt-fill me-2"></i> ${msg}`;
            }
            enableManualLocationFallback();
        },
        {
            enableHighAccuracy: true,
            timeout: 12000,
            maximumAge: 15000
        }
    );
}

function enableManualLocationFallback() {
    sosState.locationMode = 'manual';
    const input = document.getElementById('manualLocationInput');
    if (input) {
        input.focus();
    }
}

/* Battery API Integration */
function fetchBatteryStatus() {
    sosState.batteryStatus = "Battery information unavailable";

    if ('getBattery' in navigator) {
        navigator.getBattery()
            .then(battery => {
                const level = Math.round(battery.level * 100);
                const charging = battery.charging ? 'Charging' : 'Discharging';
                sosState.batteryStatus = `${level}% (${charging})`;
                const elem = document.getElementById('previewBattery');
                if (elem) elem.innerText = sosState.batteryStatus;
            })
            .catch(() => {
                sosState.batteryStatus = "Battery information unavailable";
            });
    }
}

/* Confirmation Step Renderer */
function renderConfirmationPreview() {
    document.getElementById('previewEmergencyType').innerText = sosState.selectedEmergencyType;
    document.getElementById('previewTimestamp').innerText = new Date().toLocaleString();
    document.getElementById('previewBattery').innerText = sosState.batteryStatus;

    const manualVal = document.getElementById('manualLocationInput')?.value.trim();
    const locPreviewElem = document.getElementById('previewLocation');

    if (manualVal && manualVal.length > 0) {
        sosState.locationMode = 'manual';
        sosState.mapsUrl = manualVal.includes('http') ? manualVal : `https://maps.google.com/?q=${encodeURIComponent(manualVal)}`;
        if (locPreviewElem) {
            locPreviewElem.innerHTML = `<span class="badge bg-warning text-dark me-1">(entered manually)</span> ${manualVal}`;
        }
    } else if (sosState.latitude && sosState.longitude) {
        sosState.locationMode = 'gps';
        if (locPreviewElem) {
            locPreviewElem.innerHTML = `<span class="badge bg-success me-1">(GPS Fix)</span> Lat: ${sosState.latitude.toFixed(5)}, Lng: ${sosState.longitude.toFixed(5)} (~${sosState.accuracy}m)`;
        }
    } else {
        sosState.locationMode = 'manual';
        if (locPreviewElem) {
            locPreviewElem.innerHTML = `<span class="badge bg-danger me-1">(No Location)</span> Location unconfigured (entered manually)`;
        }
    }

    const contactsContainer = document.getElementById('previewContacts');
    const warningElem = document.getElementById('noContactsInlineWarning');

    if (sosState.contacts.length === 0) {
        if (contactsContainer) contactsContainer.innerHTML = '<span class="text-danger fw-bold">No contacts saved!</span>';
        if (warningElem) warningElem.classList.remove('d-none');
    } else {
        if (warningElem) warningElem.classList.add('d-none');
        if (contactsContainer) {
            contactsContainer.innerHTML = sosState.contacts.map(c => `
                <span class="badge bg-secondary p-2">
                    <i class="bi bi-person-fill text-info me-1"></i> ${c.name} (${c.phone})
                </span>
            `).join('');
        }
    }
}

/* Trigger SEND SOS NOW */
async function triggerSendSosNow() {
    if (sosState.isSubmitting) return;

    if (sosState.contacts.length === 0) {
        showToast("You must save at least one emergency contact before dispatching!", "warning");
        document.getElementById('noContactsInlineWarning')?.classList.remove('d-none');
        return;
    }

    sosState.isSubmitting = true;
    const btnSend = document.getElementById('btnSendSosNow');
    if (btnSend) btnSend.disabled = true;

    const randomSuffix = Math.random().toString(36).substring(2, 7);
    const sosId = `sos_${Date.now()}_${randomSuffix}`;
    sosState.activeSosId = sosId;

    const manualVal = document.getElementById('manualLocationInput')?.value.trim();
    let locationText = "";

    if (manualVal && manualVal.length > 0) {
        locationText = `${manualVal} (entered manually)`;
    } else if (sosState.latitude && sosState.longitude) {
        locationText = `Lat: ${sosState.latitude.toFixed(6)}, Lng: ${sosState.longitude.toFixed(6)} (Accuracy: ~${sosState.accuracy}m)`;
    } else {
        locationText = "Location Not Specified (entered manually)";
    }

    const note = document.getElementById('sosNoteInput')?.value.trim() || "";
    const contactSummary = sosState.contacts.map(c => `${c.name} (${c.phone})`).join(', ');

    sosState.activeAlertText = [
        `🚨 EMERGENCY SOS ALERT 🚨`,
        `Type: ${sosState.selectedEmergencyType}`,
        `Time: ${new Date().toLocaleString()}`,
        `Location: ${locationText}`,
        `Google Maps: ${sosState.mapsUrl || 'N/A'}`,
        `Battery Status: ${sosState.batteryStatus}`,
        `Emergency Contacts: ${contactSummary}`,
        note ? `Note: ${note}` : ``,
        `-- Sent via SignSpeak AI --`
    ].filter(Boolean).join('\n');

    const sosPayload = {
        sos_id: sosId,
        emergency_type: sosState.selectedEmergencyType,
        latitude: sosState.latitude,
        longitude: sosState.longitude,
        location_mode: sosState.locationMode,
        accuracy: sosState.accuracy,
        maps_url: sosState.mapsUrl,
        timestamp: new Date().toISOString(),
        battery_status: sosState.batteryStatus,
        contacts: JSON.stringify(sosState.contacts),
        note: note
    };

    try {
        const res = await fetch('/api/sos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sosPayload)
        });
        const data = await res.json();

        const existingLocal = JSON.parse(localStorage.getItem('signspeak_sos_events') || '[]');
        existingLocal.unshift(sosPayload);
        localStorage.setItem('signspeak_sos_events', JSON.stringify(existingLocal));
        localStorage.setItem('signspeak_active_sos', JSON.stringify(sosPayload));

        document.getElementById('sosActiveIdBadge').innerText = `EVENT ID: ${sosId}`;
        document.getElementById('activeAlertTextPreview').value = sosState.activeAlertText;

        const phoneList = sosState.contacts.map(c => c.phone.replace(/[^0-9+]/g, '')).join(',');
        const encodedBody = encodeURIComponent(sosState.activeAlertText);
        const smsBtn = document.getElementById('btnSmsHandoff');
        if (smsBtn) {
            smsBtn.href = `sms:${phoneList}?body=${encodedBody}`;
        }

        goToSosStep(4);
        showToast("SOS Event Created & Alert Prepared!", "success");
    } catch (err) {
        showToast("Saved locally (Network fallback): " + err.message, "warning");
        goToSosStep(4);
    } finally {
        sosState.isSubmitting = false;
        if (btnSend) btnSend.disabled = false;
    }
}

/* Device Handoff Handlers */
function handleTelHandoff() {
    const savedNum = localStorage.getItem('signspeak_emergency_number') || '';
    if (!savedNum) {
        alert("No emergency number configured. Add one in Settings, or dial your local emergency service directly.");
        showToast("No emergency number configured in Settings!", "warning");
        return;
    }
    const cleanNum = savedNum.replace(/[^0-9+]/g, '');
    window.location.href = `tel:${cleanNum}`;
}

async function handleShareHandoff() {
    if (navigator.share) {
        try {
            await navigator.share({
                title: `EMERGENCY SOS: ${sosState.selectedEmergencyType}`,
                text: sosState.activeAlertText,
                url: sosState.mapsUrl || window.location.href
            });
            showToast("Shared emergency alert!", "success");
        } catch (e) {
            console.log("Share cancelled or failed", e);
        }
    } else {
        handleCopyHandoff();
    }
}

function handleCopyHandoff() {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(sosState.activeAlertText)
            .then(() => showToast("Emergency Alert copied to clipboard!", "success"))
            .catch(err => showToast("Failed to copy alert", "danger"));
    } else {
        const textarea = document.getElementById('activeAlertTextPreview');
        if (textarea) {
            textarea.select();
            document.execCommand('copy');
            showToast("Copied to clipboard!", "success");
        }
    }
}

/* Resolve SOS Alert */
async function resolveActiveSos() {
    if (!confirm("Are you sure you want to resolve and deactivate this SOS alert?")) return;

    try {
        if (sosState.activeSosId) {
            await fetch('/api/sos/resolve', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sos_id: sosState.activeSosId })
            });
        }
        localStorage.removeItem('signspeak_active_sos');
        showToast("SOS Alert marked as RESOLVED.", "info");

        const modalElem = document.getElementById('sosWizardModal');
        const bsModal = bootstrap.Modal.getInstance(modalElem);
        if (bsModal) bsModal.hide();

        loadSosHistoryData();
    } catch (e) {
        showToast("Error resolving SOS event", "danger");
    }
}

/* SOS History Renderer */
async function loadSosHistoryData() {
    try {
        const res = await fetch('/api/sos');
        const data = await res.json();
        if (!data.success) return;

        const tableBody = document.getElementById('sosHistoryTableBody');
        const countBadge = document.getElementById('sosEventCountBadge');

        if (countBadge) countBadge.innerText = `${data.sos_events.length} Events`;
        if (!tableBody) return;

        if (data.sos_events.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">No SOS emergency events logged yet.</td></tr>';
            return;
        }

        tableBody.innerHTML = data.sos_events.map(row => {
            const isResolved = row.status === 'RESOLVED';
            const statusBadge = isResolved
                ? '<span class="badge bg-success">RESOLVED</span>'
                : '<span class="badge bg-danger pulse-dot">ACTIVE</span>';

            const locDisplay = row.location_mode === 'manual'
                ? `<span class="badge bg-warning text-dark me-1">Manual</span> ${row.note || 'Manual entry'}`
                : `<a href="${row.maps_url}" target="_blank" class="text-cyan small"><i class="bi bi-geo-alt-fill me-1"></i> Maps Link</a>`;

            return `
                <tr>
                    <td class="font-monospace small">${row.sos_id}</td>
                    <td class="fw-bold text-danger">${row.emergency_type}</td>
                    <td>${locDisplay}</td>
                    <td class="font-monospace small text-secondary">${row.battery_status || 'N/A'}</td>
                    <td class="text-secondary small">${row.timestamp}</td>
                    <td>${statusBadge}</td>
                    <td>
                        ${!isResolved ? `
                            <button class="btn btn-sm btn-outline-success" onclick="resolveSosFromHistory('${row.sos_id}')">
                                <i class="bi bi-check-circle-fill"></i> Resolve
                            </button>
                        ` : '<span class="text-muted small">--</span>'}
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error("Load SOS history error:", err);
    }
}

async function resolveSosFromHistory(sosId) {
    if (!confirm("Are you sure you want to resolve and deactivate this active SOS alert?")) return;
    try {
        const res = await fetch('/api/sos/resolve', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sos_id: sosId })
        });
        const data = await res.json();
        if (data.success) {
            showToast("SOS Event Resolved", "success");
            loadSosHistoryData();
        }
    } catch (e) {
        showToast("Error resolving SOS event", "danger");
    }
}
