"use strict";
let video = document.getElementById("video");
let overlay = document.getElementById("overlay");
let ctx = overlay.getContext("2d");
let session = null;
let detections = [];
let voice = true;
let lastSpeech = "";
let lastSpeechTime = 0;
let lastPersonCapture = 0;
const status = document.getElementById("status");
const YOLO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
// CAMERA
let facingMode = "environment";
let currentStream = null;
async function startCamera() {
    try {
        if (currentStream) {
            currentStream.getTracks().forEach(t => t.stop());
        }
        currentStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: {
                    ideal: facingMode
                },
                width: {
                    ideal: 640
                },
                height: {
                    ideal: 640
                }
            },
            audio: false
        });
        video.srcObject = currentStream;
        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });
        await video.play();
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        status.innerHTML = "Camera: " + facingMode;
    } catch (err) {
        console.error(err);
        status.innerHTML = "Camera permission error";
    }
}
// LOAD MODEL
async function loadAI() {
    try {
        session = await ort.InferenceSession.create("models/yolov8n.onnx");
        console.log("Input:", session.inputNames);
        console.log("Output:", session.outputNames);
        status.innerHTML = "✅ Ready";
        detect();
    } catch (err) {
        console.error(err);
        status.innerHTML = "AI loading failed";
    }
}
// PREPROCESS IMAGE
function preprocess() {
    let canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 640;
    let c = canvas.getContext("2d");
    c.drawImage(video, 0, 0, 640, 640);
    let pixels = c.getImageData(0, 0, 640, 640).data;
    let size = 640 * 640;
    let input = new Float32Array(3 * size);
    for (let i = 0; i < pixels.length; i += 4) {
        input[i / 4] = pixels[i] / 255;
        input[size + i / 4] = pixels[i + 1] / 255;
        input[size * 2 + i / 4] = pixels[i + 2] / 255;
    }
    return input;
}
// DETECTION LOOP
async function detect() {
    if (!session || video.readyState !== 4) {
        requestAnimationFrame(detect);
        return;
    }
    let input = preprocess();
    let tensor = new ort.Tensor("float32", input,
        [1, 3, 640, 640]);
    let feeds = {};
    feeds[session.inputNames[0]] = tensor;
    let output = await session.run(feeds);
    process(output);
    requestAnimationFrame(detect);
}
// YOLO DECODER
function decodeYOLO(output) {
    let tensor = output[session.outputNames[0]];
    let data = tensor.data;
    let detections = [];
    let channels = 84;
    let boxes = 8400;
    for (let i = 0; i < boxes; i++) {
        let cx = data[i];
        let cy = data[boxes + i];
        let w = data[boxes * 2 + i];
        let h = data[boxes * 3 + i];
        let maxScore = 0;
        let classID = 0;
        for (let c = 4; c < channels; c++) {
            let score = data[c * boxes + i];
            if (score > maxScore) {
                maxScore = score;
                classID = c - 4;
            }
        }
        if (maxScore > 0.45) {
            detections.push({
                label: YOLO_CLASSES[classID],
                confidence: maxScore,
                x: (cx - w / 2) / 640 * overlay.width,
                y: (cy - h / 2) / 640 * overlay.height,
                w: w / 640 * overlay.width,
                h: h / 640 * overlay.height
            });
        }
    }
    return detections;
}
// PROCESS
function process(output) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    detections = decodeYOLO(output);
    drawBoxes();
    let objects = detections.map(x => x.label);
    document.getElementById("objects").innerHTML = objects.length ? [...new Set(objects)].join(", ") : "Searching...";
   if (objects.length) {
        let uniqueObjects = [...new Set(objects)];
        speak("I see " + uniqueObjects.join(", "));
        updateChart(objects.length);
        // AUTO SAVE WHEN PERSON DETECTED
        if (uniqueObjects.includes("person")) {
            autoCapturePerson();
        }
    }
}
// DRAW BOXES
function drawBoxes() {
    ctx.lineWidth = 3;
    ctx.font = "18px Arial";
    detections.forEach(obj => {
        ctx.strokeRect(obj.x, obj.y, obj.w, obj.h);
        ctx.fillText(obj.label, obj.x, obj.y - 5);
    });
}
// CAMERA SWITCH
async function switchCamera() {
    facingMode = facingMode === "environment" ? "user" : "environment";
    await startCamera();
}
// VOICE
function speak(text) {
    if (!voice) return;
    let now = Date.now();
    if (text === lastSpeech && now - lastSpeechTime < 5000) return;
    lastSpeech = text;
    lastSpeechTime = now;
    speechSynthesis.cancel();
    let msg = new SpeechSynthesisUtterance(text);
    msg.rate = 0.9;
    speechSynthesis.speak(msg);
}

function toggleVoice() {
    voice = !voice;
    let btn = document.querySelector("button[onclick='toggleVoice()']");
    if (voice) {
        btn.innerHTML = "🔊 Voice ON";
        speak("Voice enabled");
    } else {
        btn.innerHTML = "🔇 Voice OFF";
        speechSynthesis.cancel();
    }
}
// SCREENSHOT
function capture() {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    let link = document.createElement("a");
    link.download = "vision.jpg";
    link.href = canvas.toDataURL();
    link.click();
}

function autoCapturePerson() {
    let now = Date.now();
    // wait 10 seconds before capturing again
    if (now - lastPersonCapture < 10000) {
        return;
    }
    lastPersonCapture = now;
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let c = canvas.getContext("2d");
    c.drawImage(video, 0, 0);
    let link = document.createElement("a");
    link.download = "person_" + Date.now() + ".jpg";
    link.href = canvas.toDataURL("image/jpeg");
    link.click();
    speak("Person image saved");
}
// DESCRIPTION
function assistantMode() {
    let objects = detections.map(x => x.label);
    if (objects.length === 0) {
        speak("I cannot see any objects yet");
        return;
    }
    let unique = [...new Set(objects)];
    let text = "I can see ";
    if (unique.length === 1) {
        text += "a " + unique[0];
    } else {
        text += unique.slice(0, -1).join(", ");
        text += " and " + unique[unique.length - 1];
    }
    text += ".";
    speak(text);
    document.getElementById("objects").innerHTML = text;
}
// CHART
let chart = null;
if (document.getElementById("chart")) {
    chart = new Chart(document.getElementById("chart"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Objects",
                data: []
            }]
        }
    });
}

function updateChart(value) {
    if (!chart) return;
    chart.data.labels.push(new Date().toLocaleTimeString());
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.update();
}
// START
async function init() {
    await startCamera();
    await loadAI();
}
init();
// SERVICE WORKER
if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("service-worker.js");
}
