package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;

import java.util.stream.IntStream;

public record Adjustments(double[][][] adjustmentsWeight, double[][] adjustmentsBias) {

    public void adjust(Network network) {
        IntStream.range(0, network.denseLayers.length).parallel().forEach(layer -> {
            DenseNeuron[] neurons = network.denseLayers[layer].neurons;
            for (int neuron = 0; neuron < neurons.length; neuron++) {
                Connection[] conns = neurons[neuron].incoming;
                for (int conn = 0; conn < conns.length; conn++) {
                    conns[conn].weight -= adjustmentsWeight[layer][neuron][conn];
                }
                neurons[neuron].bias -= adjustmentsBias[layer][neuron];
            }
        });
    }

}
