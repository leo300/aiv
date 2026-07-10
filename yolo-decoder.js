const COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "cell phone", "laptop", "bottle"];

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function decodeYOLO(output, threshold = 0.45) {
    let results = [];
    /*
    YOLOv8 output:

    [1,84,8400]

    84 =
    4 box values
    +
    80 classes

    */
    let data = output.data;
    let boxes = output.dims[2];
    for (let i = 0; i < boxes; i++) {
        let confidence = data[4 * boxes + i];
        if (confidence < threshold) continue;
        let maxClass = 0;
        let maxScore = 0;
        for (let c = 0; c < 80; c++) {
            let score = data[(5 + c) * boxes + i];
            if (score > maxScore) {
                maxScore = score;
                maxClass = c;
            }
        }
        results.push({
            label: COCO_CLASSES[maxClass],
            confidence: confidence * maxScore
        });
    }
    return results;
}
