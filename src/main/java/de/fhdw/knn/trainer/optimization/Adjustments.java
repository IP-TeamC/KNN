package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;

import java.util.stream.IntStream;

/**
 * Kapselt die notwendigen Anpassungen (Gradienten der Verlustfunktion mit Berücksichtigung der Learning Rate)
 * zur Anpassung des Netzwerks
 *
 * @param adjustmentsWeight außen Layer - Neuron - eingehendes Neuron innen
 * @param adjustmentsBias   außen Layer - Neuron innen
 */
public record Adjustments(double[][][] adjustmentsWeight, double[][] adjustmentsBias) {

    /**
     * Passt das Netzwerk entsprechend an (Subtraktion der Gradienten)
     *
     * @param network anzupassendes Netzwerk
     */
    public void adjust(Network network) {
        IntStream.range(0, network.denseLayers.length).parallel().forEach(layer -> {
            AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;
            for (int neuron = 0; neuron < neurons.length; neuron++) {
                Connection[] conns = neurons[neuron].incoming;
                for (int conn = 0; conn < conns.length; conn++) {
                    conns[conn].weight -= adjustmentsWeight[layer][neuron][conn];
                }
                if (neurons[neuron] instanceof DenseNeuron dn) {
                    dn.bias -= adjustmentsBias[layer][neuron];
                }
            }
        });
    }

}
