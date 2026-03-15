package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.Network;

/**
 * Daten-Objekt Arrays für die Ausgaben/Aktivierungen und deren Ableitungen aller Neuronen des Netzwerks
 */
public class OutputsDerived {

    /**
     * Ausgabe/Aktivierung aller Neuronen eines Netzwerks (außen Layer - innen Neuron im Layer)
     */
    public final double[][] output;
    /**
     * Ableitungen der Aktivierungen aller Neuronen eines Netzwerks (außen Layer - innen Neuron im Layer)
     */
    public final double[][] derived;

    /**
     * Erzeugt ein neues OutputsDerived-Objekt
     *
     * @param output  Ausgabe/Aktivierung aller Neuronen eines Netzwerks (außen Layer - innen Neuron im Layer)
     * @param derived Ableitungen der Aktivierungen aller Neuronen eines Netzwerks (außen Layer - innen Neuron im Layer)
     */
    public OutputsDerived(double[][] output, double[][] derived) {
        this.output = output;
        this.derived = derived;
    }

    /**
     * Liefert die Ausgaben des Output Layers zurück
     *
     * @return Ausgaben des Output Layers
     */
    public double[] lastOutput() {
        return output[output.length - 1];
    }

    /**
     * Generiert ein leeres OutputsDerived-Objekt mit korrekt allokierten Arrays für die Ausgaben/Aktivierungen und deren Ableitungen aller Neuronen des Netzwerks
     *
     * @param network Netzwerk zu dessen Struktur das Objekt erzeugt wird
     * @return leeres OutputsDervied-Objekt mit bereits passend allokierten Arrays
     */
    public static OutputsDerived generateEmpty(Network network) {
        double[][] output = new double[network.denseLayers.length][];
        double[][] derived = new double[network.denseLayers.length][];
        for (int layer = 0; layer < network.denseLayers.length; layer++) {
            output[layer] = new double[network.denseLayers[layer].neurons.length];
            derived[layer] = new double[network.denseLayers[layer].neurons.length];
        }
        return new OutputsDerived(output, derived);
    }

}
