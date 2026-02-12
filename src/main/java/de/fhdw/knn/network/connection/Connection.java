package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.neuron.Neuron;

public class Connection {

    public final boolean guard;
    public double weight;
    public Neuron inputNeuron;

    public Connection(double weight) {
        this.guard = true;
        this.weight = weight;
    }

    public Connection(double weight, boolean guard) {
        this.guard = guard;
        this.weight = weight;
    }

}
