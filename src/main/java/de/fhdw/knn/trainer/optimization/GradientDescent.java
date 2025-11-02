package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.util.FutureUtil;

import java.util.stream.IntStream;

public class GradientDescent implements OptimizationFunction {

    private final Network network;
    private final LossFunction lossFunction;
    private final double learningRate;
    private NetworkPaths networkPaths;

    public GradientDescent(Network network, LossFunction lossFunction, double learningRate) {
        this.network = network;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    @Override
    public void init() {
        networkPaths = new NetworkPaths(network);
    }

    @Override
    public Adjustments compute(double[] input, double[] output) {
        double[] predictions = network.feedForward(input);
        double derivedLoss = lossFunction.derivedLoss(output, predictions);
        double adjustmentBase = learningRate * derivedLoss;

        double[][][] adjustmentsWeight = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];

        adjust(adjustmentBase, adjustmentsWeight, adjustmentsBias);
        return new Adjustments(adjustmentsWeight, adjustmentsBias);
    }

    private void adjust(double adjustmentBase, double[][][] adjustmentsWeight, double[][] adjustmentsBias) {
        int outputLayer = network.denseLayers.length - 1;
        DenseNeuron[] outputNeurons = network.denseLayers[outputLayer].neurons;

        IntStream.range(0, network.denseLayers.length).parallel().forEach(layer -> {
            DenseNeuron[] startNeurons = network.denseLayers[layer].neurons;
            adjustmentsWeight[layer] = new double[startNeurons.length][];
            adjustmentsBias[layer] = new double[startNeurons.length];

            FutureUtil.partitionedExecution(6, startNeurons.length, (first, next) -> {
                for (int start = first; start < next; start++) {
                    Connection[] conns = startNeurons[start].incoming;
                    adjustmentsWeight[layer][start] = new double[conns.length];

                    double adjustmentStart = adjustmentBase;
                    if (layer != outputLayer) {
                        adjustmentStart *= startNeurons[start].outputDerivedCache;
                    }

                    for (int output = 0; output < outputNeurons.length; output++) {
                        double adjustment = adjustmentStart
                                * outputNeurons[output].outputDerivedCache
                                * networkPaths.pathsAdjustment(layer, start, output);
                        adjustmentsBias[layer][start] += adjustment;

                        for (int conn = 0; conn < conns.length; conn++) {
                            double adjustmentWeight = adjustment * conns[conn].inputNeuron.output();
                            adjustmentsWeight[layer][start][conn] += adjustmentWeight;
                        }
                    }
                }
            });
        });
    }

}
