package de.fhdw.knn.network.neuron;

/**
 * Daten-Objekt
 * @param output Ausgabe/Aktivierung aller Neuronen eines Netzwerks (außen Layer - innen Neuron im Layer)
 * @param derived Ableitungen der Aktivierungen aller Neuronen eines Netzwerks (außen Layer - innen Neuron im Layer)
 */
public record OutputsDerived(double[][] output, double[][] derived) {

    /**
     * Liefert die Ausgaben des Output Layers zurück
     */
    public double[] lastOutput() {
        return output[output.length - 1];
    }

}
