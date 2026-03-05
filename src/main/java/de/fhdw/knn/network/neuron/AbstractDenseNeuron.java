package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.connection.Connection;

/**
 * Abstraktes Neuron im Hidden oder Output Layer.
 * Kann entweder ein {@link DenseNeuron} oder {@link SuperNeuron} sein.
 */
public abstract class AbstractDenseNeuron extends Neuron {

    /**
     * Enthält alle eingehenden Verbindungen.<br>
     * Die Connection mit Index i entspricht der Verbindung vom Neuron i des vorherigen Layers.
     */
    public Connection[] incoming;

    /**
     * Berechnet die Ausgabe/Aktivierung des Neurons für die gegebenen (noch ungewichteten) Eingaben in dieses Neuron sowie der Ableitung
     *
     * @param input Ausgabe/Aktivierung aller Neuronen des vorherigen Layers als Eingabe in dieses Neuron (noch ungewichtet)
     * @return Ausgabe/Aktivierung dieses Neurons und die Ableitung der Aktivierung
     */
    public abstract OutputDerived compute(double[] input);

}
