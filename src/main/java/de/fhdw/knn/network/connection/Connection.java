package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.neuron.Neuron;

/**
 * Verbindung von einem Neuron zu dem Neuron, das diese Connection besitzt
 */
public class Connection {

    /**
     * Setzt das Gewicht dieser Verbindung im Feedforward statisch auf 0
     */
    public boolean guard;
    /**
     * Gewicht dieser Verbindung (zur Bildung der gewichteten Summe aller eingehenden Verbindungen)
     */
    public double weight;
    /**
     * Mit dieser Verbindung eingehendes Neuron
     */
    public Neuron inputNeuron;

    /**
     * Erstellt eine Verbindung (guard = true) mit einem Gewicht.
     * <strong>Das {@link Connection#inputNeuron} muss anschließend gesetzt werden!</strong>
     *
     * @param weight Gewicht der Verbindung
     */
    public Connection(double weight) {
        this.guard = true;
        this.weight = weight;
    }

    /**
     * Erstellt eine Verbindung mit einem Gewicht und gewähltem Guard.
     * <strong>Das {@link Connection#inputNeuron} muss anschließend gesetzt werden!</strong>
     *
     * @param guard  Guard der Verbindung (setzt Gewicht statisch auf 0)
     * @param weight Gewicht der Verbindung
     */
    public Connection(double weight, boolean guard) {
        this.guard = guard;
        this.weight = weight;
    }

}
