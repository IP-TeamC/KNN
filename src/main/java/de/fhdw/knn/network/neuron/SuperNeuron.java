package de.fhdw.knn.network.neuron;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;

public class SuperNeuron extends AbstractDenseNeuron {

    public final Network network;

    /**
     * ACHTUNG! Nur für Import!
     *
     * @param network Das Netzwerk muss bereits einen Adapter enthalten!
     */
    public SuperNeuron(Network network) {
        this.network = network;
    }

    public SuperNeuron(Network network, double[] adapterBias, double[][] adapterWeights) {
        if (network.denseLayers[network.denseLayers.length - 1].neurons.length != 1) {
            throw new IllegalArgumentException("invalid super neuron network output layer size: expected 1");
        }

        // (new)[AdapterInputLayer ->] ActualInputLayer -> FirstDenseLayer
        InputLayer adapterInputLayer = new InputLayer(adapterBias.length);
        DenseLayer actualInputLayer = new DenseLayer(network.inputLayer.neurons.length);

        for (int neuron = 0; neuron < network.denseLayers[0].neurons.length; neuron++) {
            DenseNeuron dn = (DenseNeuron) network.denseLayers[0].neurons[neuron];
            for (int conn = 0; conn < dn.incoming.length; conn++) {
                dn.incoming[conn].inputNeuron = actualInputLayer.neurons[conn];
            }
        }

        for (int neuron = 0; neuron < actualInputLayer.neurons.length; neuron++) {
            DenseNeuron dn = (DenseNeuron) actualInputLayer.neurons[neuron];
            dn.activationFunction = ActivationFunction.LINEAR;
            dn.bias = adapterBias[neuron];
            dn.incoming = new Connection[adapterBias.length];
            for (int conn = 0; conn < dn.incoming.length; conn++) {
                dn.incoming[conn] = new Connection(adapterWeights[conn][neuron]);
                dn.incoming[conn].inputNeuron = adapterInputLayer.neurons[conn];
            }
        }

        DenseLayer[] denseLayers = new DenseLayer[network.denseLayers.length + 1];
        denseLayers[0] = actualInputLayer;
        System.arraycopy(network.denseLayers, 0, denseLayers, 1, network.denseLayers.length);

        this.network = new Network(adapterInputLayer, denseLayers);
    }

    @Override
    public Pair<Double, Double> compute(double[] input) {
        Pair<double[][], double[][]> output = network.feedForward(input);
        return new Pair<>(output.x[output.x.length - 1][0], output.y[output.y.length - 1][0]); // evtl. Index 0 statt output.y.length (1. oder letzter Layer?)
    }

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
