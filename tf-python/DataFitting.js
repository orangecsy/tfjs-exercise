import * as tf from '@tensorflow/tfjs';

const w = tf.variable(tf.randomUniform([1, 2]));
const b = tf.variable(tf.scalar(Math.random()));

const numIterations = 201;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [_w, _b] = [
      tf.variable(tf.tensor2d(coeff.w, [1, 2])),
      tf.scalar(coeff.b)
    ];

    const xs = tf.variable(tf.randomUniform([2, numPoints]));
    const ys = tf.matMul(_w, xs).add(_b);

    return {xs, ys};
  });
}

function predict(x) {
  return tf.tidy(() => {
    return tf.matMul(w, x).add(b);
  });
}

function loss(prediction, label) {
  const error = prediction.sub(label).square().mean();
  return error;
}

async function train(xs, ys, numIterations) {
  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const pred = predict(xs);
      return loss(pred, ys);
    });
    if (iter % 10 === 0) {
      console.log(iter, w.dataSync(), b.dataSync());
    }
    await tf.nextFrame();
  }
}

async function learnCoefficients() {
  const trueCoefficients = {w: [0.1, 0.2], b: 0.3};
  // 生成有误差的训练数据
  const trainingData = generateData(100, trueCoefficients);
  // 训练模型
  await train(trainingData.xs, trainingData.ys, numIterations); 
}

learnCoefficients();
