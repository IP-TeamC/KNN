package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.trainer.learningrate.LearningRateFunction;
import de.fhdw.knn.trainer.loss.LossFunction;

public class GradientDescent implements OptimizationFunction {

    private final LossFunction lossFunction;

    private final LearningRateFunction learningRateFunction;
    private double learningRate;

    public GradientDescent(LossFunction lossFunction, LearningRateFunction learningRateFunction) {
        this.lossFunction = lossFunction;
        this.learningRateFunction = learningRateFunction;
    }

    @Override
    public void epoch(int epoch, double previousLoss) {
        learningRate = learningRateFunction.calc(epoch, previousLoss);
    }

    @Override
    public Adjustments compute(Network network, double[] input, double[] output, int batchSize) {
        Pair<double[][], double[][]> feedForward = network.feedForward(input);
        double[][] outputs = feedForward.x;
        double[][] derived = feedForward.y;
        double[] predictions = outputs[outputs.length - 1];

        double[] derivedOutput = derived[derived.length - 1];
        double adjustmentBase = learningRate / batchSize;
        double[] adjustmentsBase = new double[output.length];
        for (int outputNeuron = 0; outputNeuron < adjustmentsBase.length; outputNeuron++) {
            adjustmentsBase[outputNeuron] = adjustmentBase * derivedOutput[outputNeuron] * lossFunction.derivedLoss(output, predictions, outputNeuron);
        }

        return compute(network, input, outputs, derived, adjustmentsBase);
    }

    private static Adjustments compute(Network network, double[] input, double[][] outputs, double[][] derived, double[] adjustmentsBase) {
        // Layer, Neuron, Connection
        double[][][] adjustmentsWeight = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];

        int outputLayer = network.denseLayers.length - 1;
        AbstractDenseNeuron[] outputNeurons = network.denseLayers[outputLayer].neurons;
        adjustmentsWeight[outputLayer] = new double[outputNeurons.length][];
        adjustmentsBias[outputLayer] = adjustmentsBase;

        for (int neuron = 0; neuron < outputNeurons.length; neuron++) {
            AbstractDenseNeuron dn = outputNeurons[neuron];
            adjustmentsWeight[outputLayer][neuron] = new double[dn.incoming.length];

            if (dn instanceof SuperNeuron sn) {
                double[] subnetInput = outputLayer == 0 ? input : outputs[outputLayer - 1];
                Pair<double[][], double[][]> subnetResults = sn.network.feedForward(subnetInput);
                double[] subAdjustmentBias = compute(sn.network, subnetInput, subnetResults.x, subnetResults.y, adjustmentsBase).adjustmentsBias()[0];

                AbstractDenseNeuron[] subLayerNeurons = sn.network.denseLayers[0].neurons;
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    double adjustmentWeight = 0;
                    for (int subNeuron = 0; subNeuron < subLayerNeurons.length; subNeuron++) {
                        adjustmentWeight += subAdjustmentBias[subNeuron] * subLayerNeurons[subNeuron].incoming[conn].weight;
                    }
                    adjustmentsBias[outputLayer][neuron] = adjustmentWeight;
                    adjustmentsWeight[outputLayer][neuron][conn] = adjustmentWeight * subnetInput[conn];
                }
            } else {
                adjustmentsWeight[outputLayer][neuron] = new double[dn.incoming.length];
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[outputLayer][neuron][conn] = adjustmentsBias[outputLayer][neuron] * outputs[outputLayer - 1][conn];
                }
            }
        }

        for (int layer = outputLayer - 1; layer >= 0; layer--) {
            double[] layerInput = layer == 0 ? input : outputs[layer - 1];
            AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;
            adjustmentsWeight[layer] = new double[neurons.length][];
            adjustmentsBias[layer] = new double[neurons.length];
            calculateAdjustments(
                    adjustmentsWeight[layer], adjustmentsBias[layer],
                    neurons, layerInput, derived[layer],
                    adjustmentsBias[layer + 1], network.denseLayers[layer + 1].neurons);
        }

        return new Adjustments(adjustmentsWeight, adjustmentsBias);
    }

    private static void calculateAdjustments(
            double[][] adjustmentsWeight, double[] adjustmentsBias,
            AbstractDenseNeuron[] neurons, double[] layerInput, double[] derived,
            double[] nextAdjustmentsBias, AbstractDenseNeuron[] nextNeurons) {
        for (int neuron = 0; neuron < neurons.length; neuron++) {
            double adjustmentBias = 0;
            for (int next = 0; next < nextNeurons.length; next++) {
                adjustmentBias += nextAdjustmentsBias[next] * nextNeurons[next].incoming[neuron].weight;
            }
            adjustmentBias *= derived[neuron];

            AbstractDenseNeuron dn = neurons[neuron];
            if (dn instanceof SuperNeuron sn) {
                Pair<double[][], double[][]> subnetResults = sn.network.feedForward(layerInput);
                double[] subAdjustmentBias = compute(sn.network, layerInput, subnetResults.x, subnetResults.y, new double[] { adjustmentBias }).adjustmentsBias()[0];

                AbstractDenseNeuron[] subLayerNeurons = sn.network.denseLayers[0].neurons;
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    double adjustmentWeight = 0;
                    for (int subNeuron = 0; subNeuron < subLayerNeurons.length; subNeuron++) {
                        adjustmentWeight += subAdjustmentBias[subNeuron] * subLayerNeurons[subNeuron].incoming[conn].weight;
                    }
                    adjustmentsBias[neuron] = adjustmentWeight;
                    adjustmentsWeight[neuron][conn] = adjustmentWeight * layerInput[conn];
                }
            } else {
                adjustmentsBias[neuron] = adjustmentBias;
                adjustmentsWeight[neuron] = new double[dn.incoming.length];
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[neuron][conn] = adjustmentBias * layerInput[conn];
                }
            }
        }
    }

}
