const path = require('path');

module.exports={
    //入口文件的配置项
    entry:{
      entry: './src/index.js'
    },
    //出口文件的配置项
    output:{
      path: path.resolve(__dirname, 'dist'),
      filename: 'bundle.js'
    }
}