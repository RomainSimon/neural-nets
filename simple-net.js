var math = require('mathjs');

function nonlin(x) {
    y = math.exp(math.subtract(0,x));
    y = y.map(function (value, index, matrix) {
      return math.add(1,value);
    });
    return math.dotDivide(1,y);
}
function nonlinderiv(x) {
    return math.dotMultiply(x, math.subtract(1,x));
}

// Initialize a layer with x,y matrix
function layer(x,y) {
    var layer = [];
    for (var i=0; i<x; i++) {
        layer.push([]);
        for (var j=0; j<y; j++) {
            layer[i][j] = 2 * Math.random() - 1;
        }
    }
    return layer;
}

// Input
var input = [[0,1,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]];

// Output
var output = [[0],
        [1],
        [1],
        [0]];


// Synapses
var syn0 = layer(3,4); // Hidden Layer
var syn1 = layer(4,1); // Output Layer


var l0,l1,l2;
var l1_error,l1_delta,l2_error,l2_delta;


// Training
for (var i=0; i<60000; i++) {
  
    // Forward propagation
    l0 = input;
    l1 = nonlin(math.multiply(l0,syn0));
    l2 = nonlin(math.multiply(l1,syn1));

    // Back propagation of errors
    l2_error = [];
    for(var j=0; j<l2.length;j++) {
        l2_error.push([output[j] - l2[j]]);
    }
    if ((i % 1000) == 0) console.log("Error (js): " + math.mean(math.abs(l2_error)) );

    l2_delta = math.dotMultiply(l2_error,nonlinderiv(l2));
    l1_error = math.multiply(l2_delta,math.transpose(syn1));
    l1_delta = math.dotMultiply(l1_error,nonlinderiv(l1));
    // console.log(l2_delta)
    // Update weights
    syn1 = math.add(syn1,math.multiply(math.transpose(l1),l2_delta));
    syn0 = math.add(syn0,math.multiply(math.transpose(l0),l1_delta));

    if (i == 60000 - 1) {
        console.log('Outbound after training (js) :')
        console.log(l2);
    }
}
