// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// use an async context to call onnxruntime functions.
async function main() {
    
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)
        const session = await ort.InferenceSession.create('./mnist-12.onnx');
        console.log('model loaded');

        const arrayLength = 784;
        var randomNumberArray = (
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,255,255,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,255,255,192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,255,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,255,224,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,255,255,192,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,255,192,128,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,255,192,0,128,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,255,255,128,0,128,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,255,224,0,0,128,255,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,255,255,192,192,192,192,248,255,224,192,192,128,192,224,224,254,252,0,0,0,0,0,0,0,0,0,128,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,252,0,0,0,0,0,0,0,0,255,255,255,255,255,224,192,192,240,255,224,192,192,128,192,224,224,254,252,0,0,0,0,0,0,0,0,0,0,255,192,128,128,0,0,0,224,255,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,255,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,255,248,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,255,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,255,254,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,255,254,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            );
        //divide random number by 255 to get a value between 0 and 1 using a map
        randomNumberArray = randomNumberArray.map(num => num / 255);
        
        const tensorA = new ort.Tensor('float32', randomNumberArray, [1,1,28,28]);

        //Get the name of the input and output node
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];
        console.log(`input name : ${inputName}`);
        console.log(`output name : ${outputName}`);


        // // // prepare feeds. use model input names as keys.
        const feeds = { Input3: tensorA};

        // // // feed inputs and run
        const results = await session.run(feeds);

        // // read from results
        const dataC = results.Plus214_Output_0.data;

        console.log(`data of result tensor 'c': ${dataC}`);
        function softmax(logits) {
            const maxLogit = Math.max(...logits);
            const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
            const sumExpLogits = expLogits.reduce((acc, expLogit) => acc + expLogit, 0);
            const softmaxProbs = expLogits.map(expLogit => expLogit / sumExpLogits);
            return softmaxProbs;
        }
        
        // Assuming dataC is an array of logits
        const softmaxProbs = softmax(dataC);
        
        // Find the index of the largest element in the softmaxProbs array
        const largestIndex = softmaxProbs.indexOf(Math.max(...softmaxProbs));
        
        console.log(`Softmax probabilities: ${softmaxProbs}`);
        console.log(`Index of the largest element: ${largestIndex}`);
}

main();