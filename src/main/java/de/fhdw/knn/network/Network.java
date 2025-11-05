package de.fhdw.knn.network;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.io.Exporter;
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

    public Network(long seed, int inputSize, int... denseLayerNeurons) {
        this.random = new Random(seed);
        this.inputLayer = new InputLayer(inputSize);
        this.denseLayers = DenseLayer.createLayers(denseLayerNeurons);
        this.connectAll();
    }

    public double[][] predict(double[][] inputs) {
        double[][] predictions = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            double[][] outputs = feedForward(inputs[i]).x;
            predictions[i] = outputs[outputs.length - 1];
        }
        return predictions;
    }

    public double[] predictSingle(double[] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double[][] outputs = feedForward(new double[]{inputs[i]}).x;
            predictions[i] = outputs[outputs.length - 1][0];
        }
        return predictions;
    }

    @Deprecated
    public double[] legacyFeedForward(double[] input) {
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

    public Pair<double[][], double[][]> feedForward(double[] input) {
        double[][] output = new double[denseLayers.length][];
        double[][] derived = new double[denseLayers.length][];

        calculateOutput(0, input, output, derived);
        for (int layer = 1; layer < denseLayers.length; layer++) {
            calculateOutput(layer, output[layer - 1], output, derived);
        }

        return new Pair<>(output, derived);
    }

    private void calculateOutput(int layer, double[] input, double[][] output, double[][] derived) {
        DenseNeuron[] neurons = denseLayers[layer].neurons;
        output[layer] = new double[neurons.length];
        derived[layer] = new double[neurons.length];

        for (int neuron = 0; neuron < neurons.length; neuron++) {
            DenseNeuron dn = denseLayers[layer].neurons[neuron];
            double weightedSum = dn.computeWeightedSum(input);
            double activation = dn.activationFunction.calc(weightedSum);
            output[layer][neuron] = activation;
            derived[layer][neuron] = dn.activationFunction.derived(weightedSum, activation);
        }
    }


    public void connectAll() {
        for (DenseNeuron denseNeuron : denseLayers[0].neurons) {
            if (denseNeuron.incoming != null) continue;
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
                if (currentNeuron.incoming != null) continue;
                currentNeuron.incoming = new Connection[prevNeurons.length];

                for (int prevNeuron = 0; prevNeuron < prevNeurons.length; prevNeuron++) {
                    Connection conn = new Connection(random.nextGaussian());
                    conn.inputNeuron = prevNeurons[prevNeuron];
                    currentNeuron.incoming[prevNeuron] = conn;
                }
            }
        }
    }

    public void export(String fileName) {
        Exporter.export(this, fileName);
    }

}
