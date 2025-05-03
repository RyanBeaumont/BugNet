let model;

async function loadModel() {
  model = await tf.loadModel('BugNet.keras');
  console.log("Model loaded.");
}
loadModel();

document.getElementById('imageUpload').addEventListener('change', function(e) {
  const ctx = document.getElementById('canvas').getContext('2d');
  const reader = new FileReader();

  reader.onload = function(event) {
    const img = new Image();
    img.onload = function() {
      // Resize to 28x28 or your model's input shape
      ctx.drawImage(img, 0, 0, 28, 28);
    };
    img.src = event.target.result;
  };

  reader.readAsDataURL(e.target.files[0]);
});

async function predictFromImage() {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }

  const canvas = document.getElementById('canvas');
  const imgTensor = tf.browser.fromPixels(canvas, 1) // 1 for grayscale 3 color
    .resizeNearestNeighbor([28, 28]) // resize if needed
    .toFloat()
    .div(255.0)
    .expandDims(0); // shape: [1, 28, 28, 1]

  const prediction = model.predict(imgTensor);
  const outputArray = await prediction.data();

  document.getElementById("output").textContent = `Prediction: [${outputArray.map(x => x.toFixed(3)).join(", ")}]`;
}
