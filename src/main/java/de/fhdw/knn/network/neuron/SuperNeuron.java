package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;

/**
 * Ein SuperNeuron enthält intern ein gesamtes KNN und funktioniert im Grunde ähnlich zu einer komplexen Aktivierungsfunktion.
 * Hiermit können vortrainierte Modelle in einem neuen Netzwerk verwendet werden, wobei ein Adapter notwendig sein kann.
 *
 * @see SuperNeuron#SuperNeuron(Network, double[], double[][])
 */
public class SuperNeuron extends AbstractDenseNeuron {

    /**
     * Das im SuperNeuron enthaltene Netzwerk (bereits inklusive Adapter)
     */
    public final Network network;

    /**
     * ACHTUNG! Nur für Import!
     *
     * @param network Das Netzwerk muss bereits einen Adapter enthalten!
     */
    public SuperNeuron(Network network) {
        this.network = network;
    }

    /**
     * Erzeuge ein Super-Neuron aus einem vorhandenen Netzwerk und setze vor dieses einen Adapter<br>
     * (neu)[AdapterInputLayer ->] ActualInputLayer (wird DenseLayer) -> FirstDenseLayer
     *
     * @param network        vorhandenes Netzwerk
     * @param adapterBias    beschreibt den Bias der Neuronen des tatsächlichen Input-Layers des vorhandenen Netzwerks, der nun zum 1. DenseLayer wird und vorher keinen Bias hatte
     * @param adapterWeights beschreibt für jedes Neuron des tatsächlichen Input-Layers des vorhandenen Netzwerks (Array außen)
     *                       die Gewichte von den eingehenden Neuronen des Adapter-Input-Layers
     *                       (hier ist jeweils nur genau das n-te Neuron des Layers vor dem Super-Neuron mit dem n-ten Neuron des Adapter-Input-Layers verbunden
     *                       und von dort aus über das hier im inneren Array angegebene Gewicht mit den tatsächlichen Input-Neuronen, die nun im DenseLayer liegen)
     */
    public SuperNeuron(Network network, double[] adapterBias, double[][] adapterWeights) {
        if (network.denseLayers[network.denseLayers.length - 1].neurons.length != 1) {
            throw new IllegalArgumentException("invalid super neuron network output layer size: expected 1");
        } else if (network.inputLayer.neurons.length != adapterBias.length || adapterBias.length != adapterWeights.length) {
            throw new IllegalArgumentException("invalid super neuron adapter or network size: requiring network input neurons = adapter bias/weights size");
        }

        // (new)[AdapterInputLayer ->] ActualInputLayer -> FirstDenseLayer
        int adapterSize = adapterWeights[0].length;
        InputLayer adapterInputLayer = new InputLayer(adapterSize);
        DenseLayer actualInputLayer = new DenseLayer(network.inputLayer.neurons.length);

        for (int denseNeuron = 0; denseNeuron < network.denseLayers[0].neurons.length; denseNeuron++) {
            AbstractDenseNeuron dn = network.denseLayers[0].neurons[denseNeuron];
            for (int actualInputNeuron = 0; actualInputNeuron < dn.incoming.length; actualInputNeuron++) {
                dn.incoming[actualInputNeuron].inputNeuron = actualInputLayer.neurons[actualInputNeuron];
            }
        }

        for (int actualInputNeuronIndex = 0; actualInputNeuronIndex < actualInputLayer.neurons.length; actualInputNeuronIndex++) {
            DenseNeuron actualInputNeuron = (DenseNeuron) actualInputLayer.neurons[actualInputNeuronIndex];
            actualInputNeuron.activationFunction = ActivationFunction.LINEAR;
            actualInputNeuron.bias = adapterBias[actualInputNeuronIndex];
            actualInputNeuron.incoming = new Connection[adapterSize];
            for (int adapterNeuron = 0; adapterNeuron < adapterSize; adapterNeuron++) {
                actualInputNeuron.incoming[adapterNeuron] = new Connection(adapterWeights[actualInputNeuronIndex][adapterNeuron]);
                actualInputNeuron.incoming[adapterNeuron].inputNeuron = adapterInputLayer.neurons[adapterNeuron];
            }
        }

        DenseLayer[] denseLayers = new DenseLayer[network.denseLayers.length + 1];
        denseLayers[0] = actualInputLayer;
        System.arraycopy(network.denseLayers, 0, denseLayers, 1, network.denseLayers.length);

        this.network = new Network(adapterInputLayer, denseLayers);
    }

    @Override
    public OutputDerived compute(double[] input) {
        double[] weightedInput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            if (incoming[i].guard) {
                weightedInput[i] = incoming[i].weight * input[i];
            }
        }

        OutputsDerived outputDerived = network.feedForward(weightedInput);
        double[][] output = outputDerived.output;
        double[][] derived = outputDerived.derived;
        return new OutputDerived(output[output.length - 1][0], derived[derived.length - 1][0]);
    }

    /**
     * Setze dieses Super-Neuron in ein vorhandenes Netzwerk ein
     *
     * @param network bereits vorhandenes Netzwerk, in dem ein Neuron durch dieses Super-Neuron ausgetauscht werden soll
     * @param layer   Index des Dense-Layers, in den das Super-Neuron eingefügt werden soll
     * @param neuron  Index des Neurons im Dense-Layer, das durch das Super-Neuron ersetzt werden soll
     */
    public void insert(Network network, int layer, int neuron) {
        incoming = network.denseLayers[layer].neurons[neuron].incoming;
        if (layer != network.denseLayers.length - 1) {
            for (AbstractDenseNeuron nextNeuron : network.denseLayers[layer + 1].neurons) {
                nextNeuron.incoming[neuron].inputNeuron = this;
            }
        }
        network.denseLayers[layer].neurons[neuron] = this;
    }

}
