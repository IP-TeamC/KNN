package de.fhdw.knn.network.neuron;

/**
 * Daten-Objekt
 *
 * @param output  Ausgabe/Aktivierung eines Neurons
 * @param derived Ableitung der Aktivierung eines Neurons
 */
public record OutputDerived(double output, double derived) {
}
