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
 * Objekte dieser Klasse sind während einer Epoche stateless und damit thread-safe!
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
     * Berechnet die Anpassungen des Netzwerks für eine Zeile des Datensatzes und überschreibt die Anpassungen im übergebenen Adjustments-Objekt.<br>
     * 1. Feed-Forward zur Berechnung der Ausgaben/Aktivierungen aller Neuronen und deren Ableitungen<br>
     * 2. Bestimmung der Anpassung des Bias der Output-Neuronen durch partielle Ableitung der Verlustfunktion nach dem jeweiligen Bias
     * und Reduzierung dieser Ableitung um den learningRate-Faktor.<br>
     * 3. Bestimmung aller weiteren Anpassungen (Gewichte der Output-Neuronen sowie aller anderen Gewichte/Bias mithilfe von {@link GradientDescent#compute(Adjustments, Network, double[], double[][], double[][])}).
     *
     * @param adjustments Anpassungen, die überschrieben werden (Verwendung als Buffer zur Reduzierung von Allokationen)
     * @param network     Netzwerk, das optimiert werden soll
     * @param input       Eingabe-Zeile aus dem Datensatz
     * @param output      Ausgabe-Zeile aus dem Datensatz
     * @param batchSize   Batch-Size für z.B. Mini-Batching (Learning Rate wird durch batchSize geteilt)
     * @see OptimizationFunction
     */
    @Override
    public void compute(Adjustments adjustments, OutputsDerived buffer, Network network, double[] input, double[] output, int batchSize) {
        network.feedForward(buffer, input);
        double[][] outputs = buffer.output;
        double[][] derived = buffer.derived;
        double[] predictions = outputs[outputs.length - 1];
        double[] derivedOutput = derived[derived.length - 1];

        double adjustmentBase = learningRate / batchSize;
        double[] adjustmentsBase = adjustments.bias[network.denseLayers.length - 1];
        for (int outputNeuron = 0; outputNeuron < adjustmentsBase.length; outputNeuron++) {
            adjustmentsBase[outputNeuron] = adjustmentBase * lossFunction.derivedLoss(output, predictions, outputNeuron) * derivedOutput[outputNeuron];
        }

        compute(adjustments, network, input, outputs, derived);
    }

    /**
     * Berechnet die Anpassungen des Netzwerks für eine Zeile des Datensatzes auf Basis vom AdjustmentsBias des Output-Layers.<br>
     * Ableitungen sind hier jeweils die Weiterführungen der Kettenregeln zur partiellen Ableitung der Verlustfunktion nach den jeweiligen Gewichten/Bias.<br>
     * 1. Berechnung der Anpassungen der Gewichte in die Output-Neuronen durch Ableitung<br>
     * 2. Iteriere über alle anderen DenseLayer von hinten nach vorne und berechne hier die Anpassungen der Gewichte/Bias mithilfe von {@link GradientDescent#calculateAdjustments(AbstractDenseNeuron[], double[], double[], double[][], double[], double[][], AbstractDenseNeuron[], double[], double[][])}<br>
     * Im Spezialfall der Super-Neuronen wird die Ableitung innerhalb des enthaltenen Netzes rekursiv weitergeführt zum jeweiligen Gewicht. Einen Bias gibt es beim Super-Neuron hingegen nicht.
     *
     * @param adjustments Anpassungen (enthalten bereits Bias für Output-Layer), die überschrieben werden (Verwendung als Buffer zur Reduzierung von Allokationen)
     * @param network     Netzwerk, das optimiert werden soll
     * @param input       Eingabe-Zeile aus dem Datensatz
     * @param outputs     Ausgaben/Aktivierungen aller Neuronen aller Layer
     * @param derived     Ableitungen der Aktivierungen aller Neuronen aller Layer
     */
    private void compute(Adjustments adjustments, Network network, double[] input, double[][] outputs, double[][] derived) {
        // Output Layer: Ermittle Eingaben und initialisiere Adjustments-Arrays
        int outputLayer = network.denseLayers.length - 1;
        double[] outputLayerInputs = outputLayer == 0 ? input : outputs[outputLayer - 1];
        AbstractDenseNeuron[] outputNeurons = network.denseLayers[outputLayer].neurons;

        // Output Layer: Berechne Adjustments für die Gewichte
        double[][] outputAdjustmentsWeight = adjustments.weight[outputLayer];
        double[] outputAdjustmentsBias = adjustments.bias[outputLayer];
        double[][] outputAdapterAdjustmentsBias = adjustments.adapterBias[outputLayer];
        for (int neuron = 0; neuron < outputNeurons.length; neuron++) {
            AbstractDenseNeuron dn = outputNeurons[neuron];

            if (dn instanceof SuperNeuron sn) {
                double[] layerInput = outputLayer == 0 ? input : outputs[outputLayer - 1];
                calculateAdjustmentsSuperNeuron(sn, layerInput,
                        outputAdjustmentsWeight[neuron], outputAdjustmentsBias[neuron], outputAdapterAdjustmentsBias[neuron]);
            } else {
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    outputAdjustmentsWeight[neuron][conn] = outputAdjustmentsBias[neuron] * outputLayerInputs[conn];
                }
            }
        }

        for (int layer = outputLayer - 1; layer >= 0; layer--) {
            // Nächster Dense Layer: Ermittle Eingaben und initialisiere Adjustments-Arrays
            double[] layerInput = layer == 0 ? input : outputs[layer - 1];
            AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;

            // Nächster Dense Layer: Berechne Adjustments für Gewichte und Bias
            int nextLayer = layer + 1;
            calculateAdjustments(
                    neurons, layerInput, derived[layer],
                    adjustments.weight[layer], adjustments.bias[layer], adjustments.adapterBias[layer],
                    network.denseLayers[nextLayer].neurons, adjustments.bias[nextLayer], adjustments.adapterBias[nextLayer]
            );
        }
    }

    /**
     * Führt die Ableitung zur Ermittlung der notwendigen Anpassungen der Gewichte/Bias für einen weiteren DenseLayer fort.
     * Zuerst wieder die Anpassung des Bias berechnet und auf Basis dessen die Anpassung der eingehenden Gewichte für jeweils jedes Neuron des Layer.
     *
     * @param neurons                         Neuronen des aktuell betrachteten Layers
     * @param layerInput                      Eingabe in diesen Layer (z.B. Ausgabe des vorherigen Layers)
     * @param layerDerived                    Ableitungen der Aktivierungen der Neuronen dieses Layers
     * @param layerAdjustmentsWeight          Anpassungen der Gewichte der Neuronen dieses Layers
     * @param layerAdjustmentsBias            Anpassungen der Bias'e der Neuronen dieses Layers
     * @param layerAdapterAdjustmentsBias     theoretische Anpassungen der Adapter-Neuronen der SuperNeuronen dieses Layers (Basis zur weiteren Berechnung für die Gewichtsanpassungen)
     * @param nextNeurons                     Neuronen des nachgelagerten Layers
     * @param nextLayerAdjustmentsBias        theoretische Anpassungen der Bias'e der Adapter-Neuronen der SuperNeuronen des nachgelagerten Layers
     * @param nextLayerAdapterAdjustmentsBias theoretische Anpassungen der Adapter-Neuronen der SuperNeuronen des nachgelagerten Layers (Basis zur weiteren Berechnung für die Gewichtsanpassungen)
     * @see GradientDescent#compute(Adjustments, Network, double[], double[][], double[][])
     */
    private void calculateAdjustments(AbstractDenseNeuron[] neurons, double[] layerInput, double[] layerDerived,
                                      double[][] layerAdjustmentsWeight, double[] layerAdjustmentsBias, double[][] layerAdapterAdjustmentsBias,
                                      AbstractDenseNeuron[] nextNeurons, double[] nextLayerAdjustmentsBias, double[][] nextLayerAdapterAdjustmentsBias) {
        for (int neuron = 0; neuron < neurons.length; neuron++) {
            double[] neuronAdjustmentsWeight = layerAdjustmentsWeight[neuron];
            AbstractDenseNeuron dn = neurons[neuron];

            // Anpassung des Bias für das Neuron berechnen
            double adjustmentBias = 0;
            for (int next = 0; next < nextNeurons.length; next++) {
                adjustmentBias +=
                        (nextNeurons[next] instanceof SuperNeuron
                                ? nextLayerAdapterAdjustmentsBias[next][neuron]
                                : nextLayerAdjustmentsBias[next])
                                * nextNeurons[next].incoming[neuron].weight;
            }
            adjustmentBias *= layerDerived[neuron];
            layerAdjustmentsBias[neuron] = adjustmentBias;

            // Anpassungen der Gewichte berechnen
            if (dn instanceof SuperNeuron sn) {
                calculateAdjustmentsSuperNeuron(sn, layerInput,
                        neuronAdjustmentsWeight, adjustmentBias, layerAdapterAdjustmentsBias[neuron]);
            } else {
                for (int conn = 0; conn < dn.incoming.length; conn++) {
                    neuronAdjustmentsWeight[conn] = adjustmentBias * layerInput[conn];
                }
            }
        }
    }

    /**
     * Berechnet die Anpassungen der Gewichte/des Bias für ein SuperNeuron durch Fortführung der Ableitung im enthaltenen Netzwerk,
     * einschließlich des linearen Einflusses des Adapters.
     *
     * @param sn                           betrachtetes SuperNeuron
     * @param layerInput                   Eingabe in diesen Layer (z.B. Ausgabe des vorherigen Layers)
     * @param neuronAdjustmentsWeight      Anpassungen der Gewichte dieses Neurons
     * @param neuronAdjustmentsBias        Anpassung des Bias dieses Neurons
     * @param neuronAdapterAdjustmentsBias theoretische Anpassungen der Adapter-Neuronen dieses SuperNeurons (Basis zur weiteren Berechnung für die Gewichtsanpassungen)
     */
    private void calculateAdjustmentsSuperNeuron(
            SuperNeuron sn, double[] layerInput,
            double[] neuronAdjustmentsWeight, double neuronAdjustmentsBias, double[] neuronAdapterAdjustmentsBias) {
        OutputsDerived subnetResults = sn.network.feedForward(layerInput);
        Adjustments subAdjustments = Adjustments.generateEmpty(sn.network);
        subAdjustments.bias[subAdjustments.bias.length - 1][0] = neuronAdjustmentsBias;
        compute(subAdjustments, sn.network, layerInput, subnetResults.output, subnetResults.derived);
        double[] subAdjustmentBias = subAdjustments.bias[0];

        AbstractDenseNeuron[] subLayerNeurons = sn.network.denseLayers[0].neurons;
        for (int adapterNeuron = 0; adapterNeuron < neuronAdapterAdjustmentsBias.length; adapterNeuron++) {
            for (int subLayerNeuron = 0; subLayerNeuron < subLayerNeurons.length; subLayerNeuron++) {
                neuronAdapterAdjustmentsBias[adapterNeuron] +=
                        subAdjustmentBias[subLayerNeuron] * subLayerNeurons[subLayerNeuron].incoming[adapterNeuron].weight;
            }
        }

        for (int conn = 0; conn < sn.incoming.length; conn++) {
            neuronAdjustmentsWeight[conn] =
                    neuronAdapterAdjustmentsBias[conn] * layerInput[conn];
        }
    }

}
