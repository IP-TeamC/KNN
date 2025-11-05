package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.trainer.loss.LossFunction;

public class GradientDescent implements OptimizationFunction {

    private final Network network;
    private final LossFunction lossFunction;
    public double learningRate;

    public GradientDescent(Network network, LossFunction lossFunction, double learningRate) {
        this.network = network;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    @Override
    public void init() {
    }

    @Override
    public Adjustments compute(double[] input, double[] output, int batchSize) {
        Pair<double[][], double[][]> feedForward = network.feedForward(input);
        double[][] outputs = feedForward.x;
        double[][] derived = feedForward.y;
        double[] predictions = outputs[outputs.length - 1];
        double adjustmentBase = learningRate * lossFunction.derivedLoss(output, predictions) / batchSize;

        // Layer, Neuron, Connection
        double[][][] adjustmentsWeight = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];

        int outputLayer = network.denseLayers.length - 1;
        DenseNeuron[] outputNeurons = network.denseLayers[outputLayer].neurons;
        adjustmentsWeight[outputLayer] = new double[outputNeurons.length][];
        adjustmentsBias[outputLayer] = new double[outputNeurons.length];

        for (int neuron = 0; neuron < outputNeurons.length; neuron++) {
            DenseNeuron dn = outputNeurons[neuron];
            double adjustmentBias = adjustmentBase * derived[outputLayer][neuron];
            adjustmentsBias[outputLayer][neuron] = adjustmentBias;

            adjustmentsWeight[outputLayer][neuron] = new double[dn.incoming.length];
            for (int conn = 0; conn < dn.incoming.length; conn++) {
                adjustmentsWeight[outputLayer][neuron][conn] = adjustmentBias * outputs[outputLayer - 1][conn];
            }
        }

        for (int layer = outputLayer - 1; layer >= 0; layer--) {
            double[] layerOutput = layer == 0 ? input : outputs[layer - 1];
            calculateAdjustments(layer, layerOutput, derived, adjustmentsWeight, adjustmentsBias);
        }

        return new Adjustments(adjustmentsWeight, adjustmentsBias);
    }

    private void calculateAdjustments(int layer, double[] prevOutput, double[][] derived, double[][][] adjustmentsWeight, double[][] adjustmentsBias) {
        int nextLayer = layer + 1;
        DenseNeuron[] neurons = network.denseLayers[layer].neurons;
        adjustmentsWeight[layer] = new double[neurons.length][];
        adjustmentsBias[layer] = new double[neurons.length];

        for (int neuron = 0; neuron < neurons.length; neuron++) {
            DenseNeuron dn = neurons[neuron];
            double adjustmentBias = 0;
            DenseNeuron[] nextNeurons = network.denseLayers[nextLayer].neurons;
            for (int prev = 0; prev < nextNeurons.length; prev++) {
                adjustmentBias += adjustmentsBias[nextLayer][prev] * nextNeurons[prev].incoming[neuron].weight;
            }
            adjustmentBias *= derived[layer][neuron];
            adjustmentsBias[layer][neuron] = adjustmentBias;

            adjustmentsWeight[layer][neuron] = new double[dn.incoming.length];
            for (int conn = 0; conn < dn.incoming.length; conn++) {
                adjustmentsWeight[layer][neuron][conn] = adjustmentBias * prevOutput[conn];
            }
        }
    }

}
