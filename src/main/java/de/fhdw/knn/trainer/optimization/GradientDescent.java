package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.OutputsDerived;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.trainer.learningrate.LearningRateFunction;
import de.fhdw.knn.trainer.loss.LossFunction;

/**
 * Beim KNN-Training üblicherweise verwendeter Optimierungsalgorithmus Gradient Descent.
 * Dieser ermittelt die Gradienten (Ableitung der Verlustfunktion nach einem Gewicht bzw. Bias),
 * passt diesen auf Basis der Learning Rate an und subtrahiert diesen von den jeweiligen Werten im Netzwerk.
 *
 * @see OptimizationFunction
 */
public class GradientDescent implements OptimizationFunction {

    /**
     * Beim Training zu optimierende Verlustfunktion
     */
    private final LossFunction lossFunction;

    /**
     * Funktion zur Bestimmung der Learning Rate jeder Epoche
     */
    private final LearningRateFunction learningRateFunction;
    /**
     * Durch die LearningRateFunction zu Beginn jeder Epoche gesetzte Learning Rate
     */
    private double learningRate;

    /**
     * Erzeugt eine Instanz des Gradient Descent Algorithmus
     *
     * @param lossFunction         Verlustfunktion für die Ermittlung des Gradienten in der Backpropagation
     * @param learningRateFunction Funktion zur Bestimmung der Learning Rate für jede Epoche
     * @see GradientDescent
     * @see OptimizationFunction
     */
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
        OutputsDerived feedForward = network.feedForward(input);
        double[][] outputs = feedForward.output();
        double[][] derived = feedForward.derived();
        double[] predictions = outputs[outputs.length - 1];
        double[] derivedOutput = derived[derived.length - 1];

        double adjustmentBase = learningRate / batchSize;
        double[] adjustmentsBase = new double[output.length];
        for (int outputNeuron = 0; outputNeuron < adjustmentsBase.length; outputNeuron++) {
            adjustmentsBase[outputNeuron] = adjustmentBase * lossFunction.derivedLoss(output, predictions, outputNeuron) * derivedOutput[outputNeuron];
        }

        return compute(network, input, outputs, derived, adjustmentsBase);
    }

    private static Adjustments compute(Network network, double[] input, double[][] outputs, double[][] derived, double[] adjustmentsBase) {
        // Layer, Neuron, Connection
        double[][][] adjustmentsWeight = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];
        double[][][] adapterAdjustmentBias = new double[network.denseLayers.length][][];

        int outputLayer = network.denseLayers.length - 1;
        double[] outputLayerInputs = outputLayer == 0 ? input : outputs[outputLayer - 1];
        AbstractDenseNeuron[] outputNeurons = network.denseLayers[outputLayer].neurons;
        adjustmentsWeight[outputLayer] = new double[outputNeurons.length][];
        adjustmentsBias[outputLayer] = adjustmentsBase;

        adapterAdjustmentBias[outputLayer] = new double[outputNeurons.length][];
        for (int neuron = 0; neuron < outputNeurons.length; neuron++) {
            AbstractDenseNeuron dn = outputNeurons[neuron];
            adjustmentsWeight[outputLayer][neuron] = new double[dn.incoming.length];

            if (dn instanceof SuperNeuron sn) {
                double[] subnetInput = outputLayer == 0 ? input : outputs[outputLayer - 1];
                OutputsDerived subnetResults = sn.network.feedForward(subnetInput);
                double[] subAdjustmentBias = compute(sn.network, subnetInput, subnetResults.output(), subnetResults.derived(),
                        adjustmentsBase).adjustmentsBias()[0];
                AbstractDenseNeuron[] subLayerNeurons = sn.network.denseLayers[0].neurons;

                adapterAdjustmentBias[outputLayer][neuron] = new double[sn.network.inputLayer.neurons.length];
                for (int adapterNeuron = 0; adapterNeuron < adapterAdjustmentBias.length; adapterNeuron++) {
                    for (int subLayerNeuron = 0; subLayerNeuron < subLayerNeurons.length; subLayerNeuron++) {
                        adapterAdjustmentBias[outputLayer][neuron][adapterNeuron] +=
                                subAdjustmentBias[subLayerNeuron] * subLayerNeurons[subLayerNeuron].incoming[adapterNeuron].weight;
                    }
                }

                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[outputLayer][neuron][conn] =
                            adapterAdjustmentBias[outputLayer][neuron][conn] * subnetInput[conn];
                }
            } else {
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[outputLayer][neuron][conn] = adjustmentsBase[neuron] * outputLayerInputs[conn];
                }
            }
        }

        for (int layer = outputLayer - 1; layer >= 0; layer--) {
            double[] layerInput = layer == 0 ? input : outputs[layer - 1];
            AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;
            adjustmentsWeight[layer] = new double[neurons.length][];
            adjustmentsBias[layer] = new double[neurons.length];
            adapterAdjustmentBias[layer] = new double[neurons.length][];
            calculateAdjustments(
                    adjustmentsWeight[layer], adjustmentsBias[layer], adapterAdjustmentBias[layer],
                    neurons, layerInput, derived[layer],
                    adjustmentsBias[layer + 1], network.denseLayers[layer + 1].neurons, adapterAdjustmentBias[layer + 1]);
        }

        return new Adjustments(adjustmentsWeight, adjustmentsBias);
    }

    private static void calculateAdjustments(
            double[][] adjustmentsWeight, double[] adjustmentsBias, double[][] adapterAdjustmentsBias,
            AbstractDenseNeuron[] neurons, double[] layerInput, double[] derived,
            double[] nextAdjustmentsBias, AbstractDenseNeuron[] nextNeurons, double[][] nextAdapterAdjustmentsBias) {
        for (int neuron = 0; neuron < neurons.length; neuron++) {
            double adjustmentBias = 0;
            for (int next = 0; next < nextNeurons.length; next++) {
                adjustmentBias +=
                        (nextAdapterAdjustmentsBias == null || nextAdapterAdjustmentsBias[next] == null
                                ? nextAdjustmentsBias[next]
                                : nextAdapterAdjustmentsBias[next][neuron])
                                * nextNeurons[next].incoming[neuron].weight;
            }
            adjustmentBias *= derived[neuron];

            AbstractDenseNeuron dn = neurons[neuron];
            adjustmentsWeight[neuron] = new double[dn.incoming.length];

            if (dn instanceof SuperNeuron sn) {
                OutputsDerived subnetResults = sn.network.feedForward(layerInput);
                double[] subAdjustmentBias = compute(sn.network, layerInput, subnetResults.output(), subnetResults.derived(),
                        new double[]{adjustmentBias}).adjustmentsBias()[0];
                AbstractDenseNeuron[] subLayerNeurons = sn.network.denseLayers[0].neurons;

                adapterAdjustmentsBias[neuron] = new double[sn.network.inputLayer.neurons.length];
                for (int adapterNeuron = 0; adapterNeuron < adapterAdjustmentsBias[neuron].length; adapterNeuron++) {
                    for (int subLayerNeuron = 0; subLayerNeuron < subLayerNeurons.length; subLayerNeuron++) {
                        adapterAdjustmentsBias[neuron][adapterNeuron] +=
                                subAdjustmentBias[subLayerNeuron] * subLayerNeurons[subLayerNeuron].incoming[adapterNeuron].weight;
                    }
                }

                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[neuron][conn] =
                            adapterAdjustmentsBias[neuron][conn] * dn.incoming[conn].weight * layerInput[conn];
                }
            } else {
                adjustmentsBias[neuron] = adjustmentBias;
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[neuron][conn] = adjustmentBias * layerInput[conn];
                }
            }
        }
    }

}
