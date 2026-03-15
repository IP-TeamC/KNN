package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;

import java.util.stream.IntStream;

/**
 * Kapselt die notwendigen Anpassungen (Gradienten der Verlustfunktion mit Berücksichtigung der Learning Rate)
 * zur Anpassung des Netzwerks
 */
// kann keine Record-Class sein, weil Array-Attribute dann nur per Getter erreichbar sind
@SuppressWarnings("ClassCanBeRecord")
public class Adjustments {

    /**
     * Anpassungen der Gewichte (außen Layer - Neuron - eingehendes Neuron innen)
     */
    public final double[][][] weight;
    /**
     * Anpassungen des Bias (außen Layer - Neuron innen)
     */
    public final double[][] bias;
    /**
     * theoretische Anpassungen der Adapter-Neuronen der Super-Neurons (außen Layer - Super-Neuron - Adapter-Neuron innerhalb Super-Neuron innen)
     */
    public final double[][][] adapterBias;

    /**
     * Erzeugt ein neues Adjustments-Objekt. Sollte nie direkt instanziiert werden, sondern immer über {@link Adjustments#generateEmpty(Network)}.
     *
     * @param weight      Anpassungen der Gewichte (außen Layer - Neuron - eingehendes Neuron innen)
     * @param bias        Anpassungen des Bias (außen Layer - Neuron innen)
     * @param adapterBias theoretische Anpassungen der Adapter-Neuronen der Super-Neurons (außen Layer - Super-Neuron - Adapter-Neuron innerhalb Super-Neuron innen)
     * @see Adjustments
     * @see Adjustments#generateEmpty(Network)
     */
    private Adjustments(double[][][] weight, double[][] bias, double[][][] adapterBias) {
        this.weight = weight;
        this.bias = bias;
        this.adapterBias = adapterBias;
    }

    /**
     * Passt das Netzwerk entsprechend an (Subtraktion der Gradienten)
     *
     * @param network anzupassendes Netzwerk
     */
    public void adjust(Network network) {
        IntStream.range(0, network.denseLayers.length)
                .forEach(layer -> adjust(network, layer));
    }

    /**
     * Passt das Netzwerk entsprechend an (Subtraktion der Gradienten).
     * Nur für einen Layer (optimierte Parallelisierung)
     *
     * @param network anzupassendes Netzwerk
     * @param layer   ausschließlich anzupassender Layer
     */
    public void adjust(Network network, int layer) {
        AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;
        for (int neuron = 0; neuron < neurons.length; neuron++) {
            Connection[] conns = neurons[neuron].incoming;
            for (int conn = 0; conn < conns.length; conn++) {
                conns[conn].weight -= weight[layer][neuron][conn];
            }
            if (neurons[neuron] instanceof DenseNeuron dn) {
                dn.bias -= bias[layer][neuron];
            }
        }
    }

    /**
     * Erzeugt leere Adjustments mit allen bereits allokierten Arrays (bis zur untersten Ebene) für das gegebene Netzwerk.
     *
     * @param network Netzwerk für dessen Struktur die Adjustments-Arrays allokiert werden sollen
     * @return leere, allokierte Adjustments zur Verwendung im Optimierungsalgorithmus
     */
    public static Adjustments generateEmpty(Network network) {
        // Layer, Neuron, Connection
        double[][][] adjustmentsWeights = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];
        double[][][] adapterAdjustmentsBias = new double[network.denseLayers.length][][];
        for (int layer = 0; layer < network.denseLayers.length; layer++) {
            AbstractDenseNeuron[] neurons = network.denseLayers[layer].neurons;
            adjustmentsWeights[layer] = new double[neurons.length][];
            adjustmentsBias[layer] = new double[neurons.length];
            adapterAdjustmentsBias[layer] = new double[neurons.length][];
            for (int neuron = 0; neuron < neurons.length; neuron++) {
                adjustmentsWeights[layer][neuron] = new double[neurons[neuron].incoming.length];
                adapterAdjustmentsBias[layer][neuron] = new double[neurons[neuron].incoming.length];
            }
        }
        return new Adjustments(adjustmentsWeights, adjustmentsBias, adapterAdjustmentsBias);
    }

}
