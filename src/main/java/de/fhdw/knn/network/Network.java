package de.fhdw.knn.network;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;

import java.util.Random;
import java.util.stream.IntStream;

public class Network {

    public final Random random;

    public InputLayer inputLayer;
    public DenseLayer[] denseLayers;

    public Network(InputLayer inputLayer, DenseLayer[] denseLayers) {
        this.random = null;
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
    }

    public Network(long seed, WeightInitializer weightInitializer, InputLayer inputLayer, DenseLayer... denseLayers) {
        this.random = new Random(seed);
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
        this.connectAll(weightInitializer);
    }

    public Network(long seed, WeightInitializer weightInitializer, int inputNeurons, DenseLayer... denseLayers) {
        this.random = new Random(seed);
        this.inputLayer = new InputLayer(inputNeurons);
        this.denseLayers = denseLayers;
        this.connectAll(weightInitializer);
    }

    public double[][] predict(double[][] inputs) {
        double[][] predictions = new double[inputs.length][];
        IntStream.range(0, inputs.length).parallel().forEach(i -> {
            double[][] outputs = feedForward(inputs[i]).x;
            predictions[i] = outputs[outputs.length - 1];
        });
        return predictions;
    }

    public double[] predictSingles(double[][] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double[][] outputs = feedForward(inputs[i]).x;
            predictions[i] = outputs[outputs.length - 1][0];
        }
        return predictions;
    }

    public double[] predictSingles(double[] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double[][] outputs = feedForward(new double[]{inputs[i]}).x;
            predictions[i] = outputs[outputs.length - 1][0];
        }
        return predictions;
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
        AbstractDenseNeuron[] neurons = denseLayers[layer].neurons;
        output[layer] = new double[neurons.length];
        derived[layer] = new double[neurons.length];

        for (int neuron = 0; neuron < neurons.length; neuron++) {
            AbstractDenseNeuron dn = denseLayers[layer].neurons[neuron];
            Pair<Double, Double> neuronOutput = dn.compute(input);
            output[layer][neuron] = neuronOutput.x;
            derived[layer][neuron] = neuronOutput.y;
        }
    }

    private void connectAll(WeightInitializer weightInitializer) {
        for (AbstractDenseNeuron denseNeuron : denseLayers[0].neurons) {
            if (denseNeuron.incoming != null) continue;
            denseNeuron.incoming = new Connection[inputLayer.neurons.length];

            for (int inputNeuron = 0; inputNeuron < inputLayer.neurons.length; inputNeuron++) {
                Connection conn = new Connection(weightInitializer.nextWeight(random, this, 0));
                conn.inputNeuron = inputLayer.neurons[inputNeuron];
                denseNeuron.incoming[inputNeuron] = conn;
            }
        }

        for (int layer = 1; layer < denseLayers.length; layer++) {
            AbstractDenseNeuron[] prevNeurons = denseLayers[layer - 1].neurons;
            AbstractDenseNeuron[] currentNeurons = denseLayers[layer].neurons;

            for (AbstractDenseNeuron currentNeuron : currentNeurons) {
                if (currentNeuron.incoming != null) continue;
                currentNeuron.incoming = new Connection[prevNeurons.length];

                for (int prevNeuron = 0; prevNeuron < prevNeurons.length; prevNeuron++) {
                    Connection conn = new Connection(weightInitializer.nextWeight(random, this, layer));
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
