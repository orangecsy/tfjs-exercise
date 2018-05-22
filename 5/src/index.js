
import * as tf from '@tensorflow/tfjs';
const echarts = require('echarts');

class CharacterTable {
  constructor(chars) {
    this.chars = chars;
    // 字符-位置index
    this.charIndices = {};
    // 位置index-字符
    this.indicesChar = {};
    this.size = this.chars.length;
    for (let i = 0; i < this.size; ++i) {
      const char = this.chars[i];
      this.charIndices[this.chars[i]] = i;
      this.indicesChar[i] = this.chars[i];
    }
  }

  // 输入questions、answers数组，输出转化的tensor
  encodeBatch(strings, maxLen) {
    const numExamples = strings.length;
    const buf = tf.buffer([numExamples, maxLen, this.size]);
    for (let i = 0; i < numExamples; ++i) {
      const str = strings[i];
      for (let j = 0; j < str.length; ++j) {
        const char = str[j];
        buf.set(1, i, j, this.charIndices[char]);
      }
    }
    return buf.toTensor().as3D(numExamples, maxLen, this.size);
  }

  decode(x, calcArgmax = true) {
    return tf.tidy(() => {
      if (calcArgmax) {
        x = x.argMax(1);
      }
      const xData = x.dataSync();
      let output = '';
      for (const index of Array.from(xData)) {
        output += this.indicesChar[index];
      }
      return output;
    });
  }
}

// digits-每个字符位数，trainingSize-训练集大小
function generateData(digits, trainingSize) {
  // 所有可选字符集
  const digitArray = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  const arraySize = digitArray.length;

  // 输出
  const output = [];
  const maxLen = digits + 1 + digits;

  // 从digitArray挑选digits个数据拼为一个数字
  const f = () => {
    let str = '';
    while (str.length < digits) {
      const index = Math.floor(Math.random() * arraySize);
      str += digitArray[index];
    }
    return Number.parseInt(str);
  };

  // 生成trainingSize组数据
  while (output.length < trainingSize) {
    const a = f();
    const b = f();

    const q = `${a}+${b}`;
    // 补空格
    const query = q + ' '.repeat(maxLen - q.length);
    let ans = (a + b).toString();
    // 补空格
    ans += ' '.repeat(digits + 1 - ans.length);
    output.push([query, ans]);
  }
  return output;
}

function convertDataToTensors(data, charTable, digits) {
  const maxLen = digits + 1 + digits;
  // data中每一项datum = [query, ans]
  const questions = data.map(datum => datum[0]);
  const answers = data.map(datum => datum[1]);
  return [
    charTable.encodeBatch(questions, maxLen),
    charTable.encodeBatch(answers, digits + 1),
  ];
}

function createAndCompileModel(hiddenSize, rnnType, digits, vocabularySize) {
  const maxLen = digits + 1 + digits;
  const model = tf.sequential();
  switch (rnnType) {
    case 'SimpleRNN':
      model.add(tf.layers.simpleRNN({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        inputShape: [maxLen, vocabularySize]
      }));
      break;
    case 'GRU':
      model.add(tf.layers.gru({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        inputShape: [maxLen, vocabularySize]
      }));
      break;
    case 'LSTM':
      model.add(tf.layers.lstm({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        inputShape: [maxLen, vocabularySize]
      }));
      break;
    default:
      break;
  }
  model.add(tf.layers.repeatVector({n: digits + 1}));
  switch (rnnType) {
    case 'SimpleRNN':
      model.add(tf.layers.simpleRNN({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true
      }));
      break;
    case 'GRU':
      model.add(tf.layers.gru({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true
      }));
      break;
    case 'LSTM':
      model.add(tf.layers.lstm({
        units: hiddenSize,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true
      }));
      break;
    default:
      break;
  }
  model.add(tf.layers.timeDistributed({
    layer: tf.layers.dense({units: vocabularySize})
  }));
  model.add(tf.layers.activation({
    activation: 'softmax'
  }));
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam',
    metrics: ['accuracy']
  });
  return model;
}

class RNN {
  constructor(digits, trainingSize, rnnType, hiddenSize) {
    // 字符集
    const chars = '0123456789+ ';
    this.charTable = new CharacterTable(chars);

    // 生成trainingSize组数据
    const data = generateData(digits, trainingSize);

    // 90%训练集，10%测试集
    const split = Math.floor(trainingSize * 0.9);
    this.trainData = data.slice(0, split);
    this.testData = data.slice(split);

    // 转为tensors，并分为训练组、测试组
    [this.trainXs, this.trainYs] = convertDataToTensors(this.trainData, this.charTable, digits);
    [this.testXs, this.testYs] = convertDataToTensors(this.testData, this.charTable, digits);
    
    // 生成模型并编译
    this.model = createAndCompileModel(hiddenSize, rnnType, digits, chars.length);
  }

  async train(iterations, batchSize, numTestExamples) {
    // 损失、准确率、每秒训练样本数，用于可视化
    const trainLossArr = [];
    const valLossArr = [];
    const trainAccuracyArr = [];
    const valAccuracyArr = [];
    const examplesPerSecArr = [];

    for (let i = 0; i < iterations; ++i) {
      // 开始计时
      const beginMs = performance.now();
      // 训练样本并测试
      const history = await this.model.fit(this.trainXs, this.trainYs, {
        epochs: 1,
        batchSize,
        validationData: [this.testXs, this.testYs],
      });
      // 时间差
      const elapsedMs = performance.now() - beginMs;
      // 每秒训练多少数据
      const examplesPerSec = this.testXs.shape[0] / (elapsedMs / 1000);
      // 训练集损失、准确率
      const trainLoss = history.history['loss'][0];
      const trainAccuracy = history.history['acc'][0];
      // 验证集损失、准确率
      const valLoss = history.history['val_loss'][0];
      const valAccuracy = history.history['val_acc'][0];

      // 输出迭代次数、训练损失、训练准确率、验证损失、验证准确滤
      document.getElementById('trainStatus').textContent =
          `Iteration ${i}: train loss = ${trainLoss.toFixed(6)}; ` +
          `train accuracy = ${trainAccuracy.toFixed(6)}; ` +
          `validation loss = ${valLoss.toFixed(6)}; ` +
          `validation accuracy = ${valAccuracy.toFixed(6)} ` +
          `(${examplesPerSec.toFixed(2)} examples/s)`;
      
      // 存入数组便于可视化
      trainLossArr.push([i, trainLoss]);
      valLossArr.push([i, valLoss]);
      trainAccuracyArr.push([i, trainAccuracy]);
      valAccuracyArr.push([i, valAccuracy]);
      examplesPerSecArr.push([i, examplesPerSec]);
      // trainLossValues.push({'epoch': i, 'loss': trainLoss});
      // valLossValues.push({'epoch': i, 'loss': valLoss});
      // trainAccuracyValues.push({'epoch': i, 'accuracy': trainAccuracy});
      // valAccuracyValues.push({'epoch': i, 'accuracy': valAccuracy});
      // examplesPerSecValues.push({'epoch': i, 'examples/s': examplesPerSec});

      // 没有展示的数据或展示的数据量不够
      if (this.testXsForDisplay == null || this.testXsForDisplay.shape[0] !== numTestExamples) {
        if (this.textXsForDisplay) {
          this.textXsForDisplay.dispose();
        }
        // 用于展示的数据
        this.testXsForDisplay = this.testXs.slice(
          [0, 0, 0],
          [numTestExamples, this.testXs.shape[1], this.testXs.shape[2]]);
      }

      const examples = [];
      const isCorrect = [];
      tf.tidy(() => {
        const predictOut = this.model.predict(this.testXsForDisplay);
        for (let j = 0; j < numTestExamples; ++j) {
          const scores = predictOut
            .slice([j, 0, 0], [1, predictOut.shape[1], predictOut.shape[2]])
            .as2D(predictOut.shape[1], predictOut.shape[2]);
          const decoded = this.charTable.decode(scores);
          examples.push(this.testData[j][0] + ' = ' + decoded);
          isCorrect.push(this.testData[j][1].trim() === decoded.trim());
        }
      });

      // 修改testExamples视图
      const examplesDiv = document.getElementById('testExamples');
      // 清空
      while (examplesDiv.firstChild) {
        examplesDiv.removeChild(examplesDiv.firstChild);
      }
      // 向页面节点添加测试样本
      for (let i = 0; i < examples.length; ++i) {
        const exampleDiv = document.createElement('div');
        exampleDiv.textContent = examples[i];
        // 为正确和错误结果的节点加上不同的类
        exampleDiv.className = isCorrect[i] ? 'answer-correct' : 'answer-wrong';
        examplesDiv.appendChild(exampleDiv);
      }
      // 迭代
      await tf.nextFrame();
    }
    // 可视化
    const lossCanvas = echarts.init(document.getElementById('lossCanvas'));
    const accuracyCanvas = echarts.init(document.getElementById('accuracyCanvas'));
    const examplesPerSecCanvas = echarts.init(document.getElementById('examplesPerSecCanvas'));
    lossCanvas.setOption({
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
        name: 'trainLoss',  
        type: 'line',
        symbol: 'none',
        data: trainLossArr  
      },{
        name: 'valLoss',
        type: 'line',
        symbol: 'none',
        data: valLossArr
      }]
    });
    accuracyCanvas.setOption({
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
        name: 'trainAccuracy',
        type: 'line',
        symbol: 'none',
        data: trainAccuracyArr
      },{
        name: 'valAccuracy',
        type: 'line',
        symbol: 'none',
        data: valAccuracyArr
      }]
    });
    examplesPerSecCanvas.setOption({
      title: {
        text: 'Examples Per Second'
      },
      xAxis: {
        type: 'value'
      },
      yAxis: {
        type: 'value'
      },
      series: {
        name: 'examplesPerSec',
        type: 'line',
        symbol: 'none',
        data: examplesPerSecArr
      }
    });
  }
}

async function runRNN() {
  // 训练按钮
  document.getElementById('trainModel').addEventListener('click', async () => {
    // 获取参数
    // 每个字符位数
    const digits = +(document.getElementById('digits')).value;
    // 训练集大小
    const trainingSize = +(document.getElementById('trainingSize')).value;
    // SimpleRNN、GRU、LSTM
    const rnnTypeSelect = document.getElementById('rnnType');
    const rnnType = rnnTypeSelect.options[rnnTypeSelect.selectedIndex].getAttribute('value');
    // 隐含层大小
    const hiddenSize = +(document.getElementById('rnnLayerSize')).value;

    // 迭代次数
    const trainIterations = +(document.getElementById('trainIterations')).value;
    // 每次训练大小
    const batchSize = +(document.getElementById('batchSize')).value;
    // 测试几组数据
    const numTestExamples = +(document.getElementById('numTestExamples')).value;

    // 生成RNN
    const demo = new RNN(digits, trainingSize, rnnType, hiddenSize);
    // 训练RNN
    await demo.train(trainIterations, batchSize, numTestExamples);
  });
}
 

runRNN();
