
var echarts = require('echarts');

const statusElement = document.getElementById('status');
const imagesElement = document.getElementById('images');

export function isTraining() {
  statusElement.innerText = 'Training...';
}

export function showTestResults(batch, predictions, labels) {
  statusElement.innerText = 'Testing...';

  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    draw(image.flatten(), canvas);

    const pred = document.createElement('div');

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }
}

const lossChart = echarts.init(document.getElementById('lossChart'));
const accuracyChart = echarts.init(document.getElementById('accuracyChart'));
export function plot(lossValues, accuracyValues) {
  lossChart.setOption({
    title: {
      text: 'Loss Values'
    },
    xAxis: {
      type: 'value'
    },
    yAxis: {
      type: 'value'
    },
    series: [{
      name: 'loss',
      type: 'line',
      data: lossValues
    }]
  });
  accuracyChart.setOption({
    title: {
      text: 'Accuracy Values'
    },
    xAxis: {
      type: 'value'
    },
    yAxis: {
      type: 'value'
    },
    series: [{
      name: 'accuracy',
      type: 'line',
      data: accuracyValues
    }]
  });
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
