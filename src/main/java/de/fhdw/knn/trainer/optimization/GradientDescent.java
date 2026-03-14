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

    /**
     * Berechnet die Anpassungen des Netzwerks für eine Zeile des Datensatzes.<br>
     * 1. Feed-Forward zur Berechnung der Ausgaben/Aktivierungen aller Neuronen und deren Ableitungen<br>
     * 2. Bestimmung der Anpassung des Bias der Output-Neuronen durch partielle Ableitung der Verlustfunktion nach dem jeweiligen Bias
     * und Reduzierung dieser Ableitung um den learningRate-Faktor.<br>
     * 3. Bestimmung aller weiteren Anpassungen (Gewichte der Output-Neuronen sowie aller anderen Gewichte/Bias mithilfe von {@link GradientDescent#compute(Network, double[], double[][], double[][], double[])}).
     *
     * @param network   Netzwerk, das optimiert werden soll
     * @param input     Eingabe-Zeile aus dem Datensatz
     * @param output    Ausgabe-Zeile aus dem Datensatz
     * @param batchSize Batch-Size für z.B. Mini-Batching (Learning Rate wird durch batchSize geteilt)
     * @return berechnete Anpassungen des Netzwerks für die übergebene Zeile des Datensatzes
     * @see OptimizationFunction
     */
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

    /**
     * Berechnet die Anpassungen des Netzwerks für eine Zeile des Datensatzes auf Basis von adjustmentsBase.<br>
     * Ableitungen sind hier jeweils die Weiterführungen der Kettenregeln zur partiellen Ableitung der Verlustfunktion nach den jeweiligen Gewichten/Bias.<br>
     * 1. Berechnung der Anpassungen der Gewichte in die Output-Neuronen durch Ableitung<br>
     * 2. Iteriere über alle anderen DenseLayer von hinten nach vorne und berechne hier die Anpassungen der Gewichte/Bias mithilfe von {@link GradientDescent#calculateAdjustments(LayerComputeValues)}<br>
     * Im Spezialfall der Super-Neuronen wird die Ableitung innerhalb des enthaltenen Netzes rekursiv weitergeführt zum jeweiligen Gewicht. Einen Bias gibt es beim Super-Neuron hingegen nicht.
     *
     * @param network         Netzwerk, das optimiert werden soll
     * @param input           Eingabe-Zeile aus dem Datensatz
     * @param outputs         Ausgaben/Aktivierungen aller Neuronen aller Layer
     * @param derived         Ableitungen der Aktivierungen aller Neuronen aller Layer
     * @param adjustmentsBase Basis-Anpassung (Anpassung des Bias aller Output-Neuronen)
     * @return berechnete Anpassungen des Netzwerks für die übergebene Zeile des Datensatzes
     */
    private static Adjustments compute(Network network, double[] input, double[][] outputs, double[][] derived, double[] adjustmentsBase) {
        // Layer, Neuron, Connection
        double[][][] adjustmentsWeight = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];
        double[][][] adapterAdjustmentBias = new double[network.denseLayers.length][][];

        // Output Layer: Ermittle Eingaben und initialisiere Adjustments-Arrays
        int outputLayer = network.denseLayers.length - 1;
        double[] outputLayerInputs = outputLayer == 0 ? input : outputs[outputLayer - 1];
        AbstractDenseNeuron[] outputNeurons = network.denseLayers[outputLayer].neurons;
        adjustmentsWeight[outputLayer] = new double[outputNeurons.length][];
        adjustmentsBias[outputLayer] = adjustmentsBase;

        // Output Layer: Berechne Adjustments für die Gewichte
        adapterAdjustmentBias[outputLayer] = new double[outputNeurons.length][];
        for (int neuron = 0; neuron < outputNeurons.length; neuron++) {
            AbstractDenseNeuron dn = outputNeurons[neuron];
            adjustmentsWeight[outputLayer][neuron] = new double[dn.incoming.length];

            if (dn instanceof SuperNeuron sn) {
                double[] layerInput = outputLayer == 0 ? input : outputs[outputLayer - 1];
                calculateAdjustmentsSuperNeuron(sn, neuron, layerInput,
                        adjustmentsBias[outputLayer][neuron], adapterAdjustmentBias[outputLayer], adjustmentsWeight[outputLayer]);
            } else {
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    adjustmentsWeight[outputLayer][neuron][conn] = adjustmentsBase[neuron] * outputLayerInputs[conn];
                }
            }
        }

        for (int layer = outputLayer - 1; layer >= 0; layer--) {
            // Nächster Dense Layer: Ermittle Eingaben und initialisiere Adjustments-Arrays
            double[] layerInput = layer == 0 ? input : outputs[layer - 1];
            AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;
            adjustmentsWeight[layer] = new double[neurons.length][];
            adjustmentsBias[layer] = new double[neurons.length];
            adapterAdjustmentBias[layer] = new double[neurons.length][];

            // Nächster Dense Layer: Berechne Adjustments für Gewichte und Bias
            calculateAdjustments(new LayerComputeValues(
                    adjustmentsWeight[layer], adjustmentsBias[layer], adapterAdjustmentBias[layer],
                    neurons, layerInput, derived[layer],
                    adjustmentsBias[layer + 1], network.denseLayers[layer + 1].neurons, adapterAdjustmentBias[layer + 1]
            ));
        }

        return new Adjustments(adjustmentsWeight, adjustmentsBias);
    }

    /**
     * Führt die Ableitung zur Ermittlung der notwendigen Anpassungen der Gewichte/Bias für einen weiteren DenseLayer fort.
     * Zuerst wieder die Anpassung des Bias berechnet und auf Basis dessen die Anpassung der eingehenden Gewichte für jeweils jedes Neuron des Layer.
     *
     * @param values Alle notwendigen Werte zur Berechnung der Anpassungen für einen Layer, siehe {@link LayerComputeValues}
     * @see GradientDescent#compute(Network, double[], double[][], double[][], double[])
     */
    private static void calculateAdjustments(LayerComputeValues values) {
        for (int neuron = 0; neuron < values.neurons.length; neuron++) {
            AbstractDenseNeuron dn = values.neurons[neuron];
            values.adjustmentsWeight[neuron] = new double[dn.incoming.length];

            // Anpassung des Bias für das Neuron berechnen
            // noinspection ExtractMethodRecommender (kleine Performance-Verbesserung)
            double adjustmentBias = 0;
            for (int next = 0; next < values.nextNeurons.length; next++) {
                adjustmentBias +=
                        (values.nextAdapterAdjustmentsBias[next] == null
                                ? values.nextAdjustmentsBias[next]
                                : values.nextAdapterAdjustmentsBias[next][neuron])
                                * values.nextNeurons[next].incoming[neuron].weight;
            }
            adjustmentBias *= values.derived[neuron];

            // Anpassungen der Gewichte berechnen
            if (dn instanceof SuperNeuron sn) {
                calculateAdjustmentsSuperNeuron(sn, neuron, values.layerInput,
                        adjustmentBias, values.adapterAdjustmentsBias, values.adjustmentsWeight);
            } else {
                values.adjustmentsBias[neuron] = adjustmentBias;
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    values.adjustmentsWeight[neuron][conn] = adjustmentBias * values.layerInput[conn];
                }
            }
        }
    }

    /**
     * Berechnet die Anpassungen der Gewichte/des Bias für ein SuperNeuron durch Fortführung der Ableitung im enthaltenen Netzwerk,
     * einschließlich des linearen Einflusses des Adapters.
     *
     * @param sn                     betrachtetes SuperNeuron
     * @param neuron                 Index des SuperNeurons im Layer
     * @param layerInput             Eingabe in diesen Layer (z.B. Ausgabe des vorherigen Layers)
     * @param adjustmentBias         berechnete Anpassung des Bias, falls es sich um ein herkömmliches DenseNeuron handeln würde (Basis zur Berechnung innerhalb des enthaltenen Netzwerks)
     * @param adapterAdjustmentsBias zu berechnende theoretische Anpassungen der Adapter-Neuronen dieses SuperNeurons (Basis zur weiteren Berechnung für die Gewichtsanpassungen)
     * @param adjustmentsWeight      zu berechnende Anpassungen einer Verbindung dieses SuperNeurons
     */
    private static void calculateAdjustmentsSuperNeuron(
            SuperNeuron sn, int neuron, double[] layerInput,
            double adjustmentBias, double[][] adapterAdjustmentsBias, double[][] adjustmentsWeight) {
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

        for (int conn = 0; conn < sn.incoming.length; conn++) {
            adjustmentsWeight[neuron][conn] =
                    adapterAdjustmentsBias[neuron][conn] * layerInput[conn];
        }
    }

    /**
     * Alle notwendigen Werte zur Berechnung der Anpassungen für einen Layer
     *
     * @param adjustmentsWeight          Array der zu berechnenden Anpassungen der Gewichte der in Neuronen dieses Layers eingehender Verbindungen
     * @param adjustmentsBias            Array der zu berechnenden Anpassungen der Bias'e der Neuronen dieses Layers
     * @param adapterAdjustmentsBias     Array der zu berechnenden theoretischen Anpassungen der Adapter-Neuronen möglicherweise enthaltener Super-Neuronen
     * @param neurons                    Neuronen des Layers
     * @param layerInput                 Eingabe in diesen Layer (z.B. Ausgabe des vorherigen Layers)
     * @param derived                    Ableitungen der Aktivierungen der Neuronen dieses Layers
     * @param nextAdjustmentsBias        bereits berechnete Anpassungen der Bias'e der Neuronen des nachgelagerten Layers (Backpropagation)
     * @param nextNeurons                Neuronen des nachgelagerten Layers
     * @param nextAdapterAdjustmentsBias bereits berechnete theoretische Anpassungen der Adapter-Neuronen möglicherweise enthaltener Super-Neuronen des nachgelagerten Layers
     */
    private record LayerComputeValues(double[][] adjustmentsWeight, double[] adjustmentsBias,
                                      double[][] adapterAdjustmentsBias,
                                      AbstractDenseNeuron[] neurons, double[] layerInput, double[] derived,
                                      double[] nextAdjustmentsBias, AbstractDenseNeuron[] nextNeurons,
                                      double[][] nextAdapterAdjustmentsBias) {
    }

}
