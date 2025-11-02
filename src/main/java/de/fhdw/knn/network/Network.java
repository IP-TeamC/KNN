package de.fhdw.knn.network;

import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;

import java.util.Random;

public class Network {

    public final Random random;

    public InputLayer inputLayer;
    public DenseLayer[] denseLayers;

    public Network(long seed, InputLayer inputLayer, DenseLayer... denseLayers) {
        this.random = new Random(seed);
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
        this.connectAll();
    }

    public Network(long seed, int inputSize, int hiddenLayers, int neuronsPerHiddenLayer, int outputNeurons) {
        this.random = new Random(seed);
        this.inputLayer = new InputLayer(inputSize);
        this.denseLayers = new DenseLayer[hiddenLayers + 1];
        for (int i = 0; i < hiddenLayers; i++) {
            this.denseLayers[i] = new DenseLayer(neuronsPerHiddenLayer);
        }
        this.denseLayers[hiddenLayers] = new DenseLayer(outputNeurons);
        this.connectAll();
    }

    public double[][] predict(double[][] inputs) {
        double[][] predictions = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = feedForward(inputs[i]);
        }
        return predictions;
    }

    public double[] predictSingle(double[][] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = feedForward(inputs[i])[0];
        }
        return predictions;
    }

    public double[] feedForward(double[] input) {
        for (int i = 0; i < input.length; i++) {
            inputLayer.neurons[i].valueCache = input[i];
        }
        for (DenseLayer denseLayer : denseLayers) {
            for (DenseNeuron denseNeuron : denseLayer.neurons) {
                denseNeuron.compute();
            }
        }

        DenseLayer outputLayer = denseLayers[denseLayers.length - 1];
        double[] output = new double[outputLayer.neurons.length];
        for (int i = 0; i < outputLayer.neurons.length; i++) {
            output[i] = outputLayer.neurons[i].outputCache;
        }
        return output;
    }

    public void connectAll() {
        for (DenseNeuron denseNeuron : denseLayers[0].neurons) {
            denseNeuron.incoming = new Connection[inputLayer.neurons.length];
            for (int inputNeuron = 0; inputNeuron < inputLayer.neurons.length; inputNeuron++) {
                Connection conn = new Connection(random.nextGaussian());
                conn.inputNeuron = inputLayer.neurons[inputNeuron];
                denseNeuron.incoming[inputNeuron] = conn;
            }
        }

        for (int layer = 1; layer < denseLayers.length; layer++) {
            DenseNeuron[] currentNeurons = denseLayers[layer].neurons;
            DenseNeuron[] prevNeurons = denseLayers[layer - 1].neurons;
            for (DenseNeuron currentNeuron : currentNeurons) {
                currentNeuron.incoming = new Connection[prevNeurons.length];
                for (int prevNeuron = 0; prevNeuron < prevNeurons.length; prevNeuron++) {
                    Connection conn = new Connection(random.nextGaussian());
                    conn.inputNeuron = prevNeurons[prevNeuron];
                    currentNeuron.incoming[prevNeuron] = conn;
                }
            }
        }
    }

}
