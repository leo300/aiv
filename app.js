let video = document.getElementById("video");
let overlay = document.getElementById("overlay");
let ctx = overlay.getContext("2d");
let session;
let detections = [];
let voice = true;
let history = [];
const status = document.getElementById("status");
const COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
// CAMERA
let facingMode = "environment";
let currentStream = null;
async function startCamera() {
    try {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
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
        await video.play();
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        status.innerHTML = "Camera: " + facingMode;
    } catch (error) {
        console.error(error);
        status.innerHTML = "Camera permission error";
    }
}
// LOAD YOLO
async function loadAI() {
    session = await ort.InferenceSession.create("models/yolov8n.onnx");
    status.innerHTML = "✅ YOLOv8 Ready";
    detect();
}
// DETECTION LOOP
async function detect() {
    let input = await preprocess();
    let tensor = new ort.Tensor("float32", input,
        [1, 3, 640, 640]);
    let output = await session.run({
        images: tensor
    });
    process(output);
    requestAnimationFrame(detect);
}

function preprocess() {
    let canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 640;
    let c = canvas.getContext("2d");
    c.drawImage(video, 0, 0, 640, 640);
    let data = c.getImageData(0, 0, 640, 640).data;
    let arr = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < data.length; i += 4) {
        arr[i / 4] = data[i] / 255;
        arr[640 * 640 + i / 4] = data[i + 1] / 255;
        arr[2 * 640 * 640 + i / 4] = data[i + 2] / 255;
    }
    return arr;
}

function process(output) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    // Real YOLO parsing goes here
    // output tensor contains boxes
    // Demo UI update
    let objects = ["person"];
    document.getElementById("objects").innerHTML = objects.join(",");
    if (voice) speak("I see " + objects.join(","));
    updateChart(objects.length);
}
// SWITCH CAMERA
async function switchCamera() {
    if (facingMode === "environment") {
        facingMode = "user";
    } else {
        facingMode = "environment";
    }
    await startCamera();
}
// VOICE
function speak(text) {
    let msg = new SpeechSynthesisUtterance(text);
    msg.rate = .9;
    speechSynthesis.speak(msg);
}

function toggleVoice() {
    voice = !voice;
}
// SCREENSHOT
function capture() {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    let a = document.createElement("a");
    a.download = "vision.jpg";
    a.href = canvas.toDataURL();
    a.click();
}
// AI DESCRIPTION
function assistantMode() {
    let detected = detections.map(x => x.label);
    if (detected.length === 0) {
        speak("I cannot see any objects yet");
        return;
    }
    let unique = [...new Set(detected)];
    let description = "I can see ";
    if (unique.length === 1) {
        description += "a " + unique[0];
    } else {
        description += unique.slice(0, -1).join(", ");
        description += " and " + unique.at(-1);
    }
    description += ".";
    speak(description);
    document.getElementById("objects").innerHTML = description;
}
// GRAPH
let chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Objects",
            data: []
        }]
    }
});

function updateChart(value) {
    chart.data.labels.push(new Date().toLocaleTimeString());
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.update();
}
async function init() {
    await startCamera();
    await loadAI();
}
init();
if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("service-worker.js");
}
