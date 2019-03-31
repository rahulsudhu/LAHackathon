/* 
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

var _ = require('lodash');
var tf = require('@tensorflow/tfjs');
//require('@tensorflow/tfjs-node');
var itradedata = require('./itradeconv.json');

var productName = itradedata.map((n) => {
    return n["Category Name"];
});
var hotName = _.uniq(productName);
var nameindex = [];
for (var i = 0; i < productName.length; i++) {
    nameindex.push(hotName.findIndex((element) => {
        return element === productName[i];
    }));
};
var leng = hotName.length;
hotName = tf.oneHot(tf.tensor1d(nameindex).toInt(), leng);


var vendorName = itradedata.map((n) => {
    return n["Vendor Name"];
});
var hotVendor = _.uniq(vendorName);
var vendorindex = [];
for (var i = 0; i < vendorName.length; i++) {
    vendorindex.push(hotVendor.findIndex((element) => {
        return element === vendorName[i];
    }));
};
var leng4 = hotVendor.length;
hotVendor = tf.oneHot(tf.tensor1d(vendorindex).toInt(), leng4);

var wareHouseName = itradedata.map((n) => {
    return n["Shipping Warehouse"];
});
var hotWarehouse = _.uniq(wareHouseName);
var wareindex = [];
for (var i = 0; i < wareHouseName.length; i++) {
    wareindex.push(hotWarehouse.findIndex((element) => {
        return element === wareHouseName[i];
    }));
};
var leng2 = hotWarehouse.length;
hotWarehouse = tf.oneHot(tf.tensor1d(wareindex).toInt(), leng2);

var month = itradedata.map((n) => {
    return n["Shipping Date"].slice(2, 6);
});
var hotmonth = _.uniq(month);
var monthindex = [];
for (var i = 0; i < month.length; i++) {
    monthindex.push(hotmonth.findIndex((element) => {
        return element === month[i];
    }));
};
var leng3 = hotmonth.length;
// console.log(monthindex);
hotmonth = tf.oneHot(tf.tensor1d(monthindex).toInt(), leng3);


var labelValues = itradedata.map((n) => {
    return (parseInt(n["Failed Cases"]) / parseInt(n["Total Cases"]));
});
labelValues = tf.tensor(labelValues);

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [leng+leng2+leng3+leng4], units: 227620, activation: 'sigmoid', kernelInitializer: 'leCunNormal'}));
//model.add(tf.layers.dense({units: leng3, activation: 'sigmoid'}));
//model.add(tf.layers.dense({units: leng4, activation: 'sigmoid'}));
//model.add(tf.layers.dense({units: 1000, activation: 'sigmoid'}));
model.add(tf.layers.dense({units: 1}));

tf.util.shuffle(hotName);
tf.util.shuffle(hotWarehouse);
tf.util.shuffle(hotmonth);

model.compile({
  optimizer: tf.train.sgd(0.01),
  loss: tf.losses.meanSquaredError,
  metrics: ['accuracy']
});

model.fit([hotName,hotWarehouse,hotVendor,hotmonth], labelValues, {
    epoch: 50,
    batchSize: 20000,
    callback: console.log("running")
}).catch((err) => {
}); 

var testname = tf.oneHot(tf.tensor1d([2]).toInt(), leng);
var testhouse = tf.oneHot(tf.tensor1d([20]).toInt(), leng2);
var testmonth = tf.oneHot(tf.tensor1d([4]).toInt(), leng4);
var testvendor = tf.oneHot(tf.tensor1d([12]).toInt(), leng3);

model.predict(tf.concat([testname.squeeze(), testhouse.squeeze(), testmonth.squeeze(), testvendor.squeeze()]).as2D(1,leng+leng2+leng3+leng4)).print();

