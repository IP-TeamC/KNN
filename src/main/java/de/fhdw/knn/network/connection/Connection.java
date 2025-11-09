package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.neuron.Neuron;

public class Connection {

    public final boolean guard = true;
    public double weight;
    public Neuron inputNeuron;

    public Connection(double weight) {
        this.weight = weight;
    }

}
