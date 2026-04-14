const videoElement = document.createElement("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const synth = window.speechSynthesis;

let sequence = [];
const SEQ_LENGTH = 20;

let dataset = [];
let currentLabel = null;

let model, labels = [];

// 🎤 VOZ
function speak(text) {
  if (synth.speaking) return;

  const msg = new SpeechSynthesisUtterance(text);
  msg.lang = "en-US";
  synth.speak(msg);

  document.getElementById("estadoVoz").innerText = "Speaking: " + text;
}

// 🎮 SET LABEL
function setLabel(label) {
  currentLabel = label;
  console.log("Recording:", label);
}

// 🤖 MEDIAPIPE
const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
  }
});

hands.setOptions({
  maxNumHands: 1,
  minDetectionConfidence: 0.8,
  minTrackingConfidence: 0.8
});

hands.onResults(results => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    drawConnectors(ctx, landmarks, HAND_CONNECTIONS);
    drawLandmarks(ctx, landmarks);

    // NORMALIZAR
    const baseX = landmarks[0].x;
    const baseY = landmarks[0].y;

    const frame = landmarks.flatMap(p => [
      p.x - baseX,
      p.y - baseY,
      p.z
    ]);

    sequence.push(frame);
    if (sequence.length > SEQ_LENGTH) sequence.shift();

    // GRABAR DATASET
    if (currentLabel && sequence.length === SEQ_LENGTH) {
      dataset.push({
        label: currentLabel,
        data: [...sequence]
      });
    }

    // PREDICCIÓN
    if (model && sequence.length === SEQ_LENGTH) {
      const input = tf.tensor([sequence]);
      const prediction = model.predict(input);
      const index = prediction.argMax(1).dataSync()[0];

      const label = labels[index];

      document.getElementById("result").innerText = label;
      speak(label);
    }

  } else {
    document.getElementById("result").innerText = "No hand detected";
  }
});

// 📷 CAMARA
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 300,
  height: 300
});

function init() {
  canvas.width = 300;
  canvas.height = 300;

  camera.start().then(() => {
    console.log("Camera started");
  }).catch(err => {
    console.error("Camera error:", err);
    document.getElementById("estadoVoz").innerText =
      "Camera error: " + err.message;
  });
}

// 🧠 TRAIN MODEL
async function trainModel() {
  labels = [...new Set(dataset.map(d => d.label))];

  const xs = tf.tensor3d(dataset.map(d => d.data));
  const ys = tf.tensor2d(
    dataset.map(d =>
      labels.map(l => (l === d.label ? 1 : 0))
    )
  );

  model = tf.sequential();

  model.add(tf.layers.lstm({
    inputShape: [SEQ_LENGTH, 63],
    units: 64
  }));

  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: labels.length, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy"
  });

  await model.fit(xs, ys, { epochs: 20 });

  console.log("Model trained");
}

// 💾 GUARDAR
async function saveModel() {
  await model.save('downloads://kinesign-model');
}
