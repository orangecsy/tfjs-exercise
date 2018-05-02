
import * as tf from '@tensorflow/tfjs';
import {MnistData} from './data';
import * as ui from './ui';

const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
  units: 10,
  kernelInitializer: 'varianceScaling',
  activation: 'softmax'
}));

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

async function train() {
  ui.isTraining();

  const lossValues = [];
  const accuracyValues = [];

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = data.nextTrainBatch(BATCH_SIZE);

    let testBatch;
    let validationData;
    if (i % TEST_ITERATION_FREQUENCY === 0) {
      testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
      validationData = [
        testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
      ];
    }

    const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
      batch.labels,
      {batchSize: BATCH_SIZE, validationData, epochs: 1}
    );
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];
    lossValues.push([i, loss]);

    if (testBatch != null) {
      accuracyValues.push([i, accuracy]);
    }

    batch.xs.dispose();
    batch.labels.dispose();
    if (testBatch != null) {
      testBatch.xs.dispose();
      testBatch.labels.dispose();
    }

    await tf.nextFrame();
  }
  ui.plot(lossValues, accuracyValues);
}

async function showPredictions() {
  const testExamples = 100;
  const batch = data.nextTestBatch(testExamples);

  tf.tidy(() => {
    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());
    
    ui.showTestResults(batch, predictions, labels);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  await load();

  await train();

  showPredictions();
}


mnist();
