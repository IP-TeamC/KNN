package de.fhdw.knn.network.neuron;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.connection.Connection;

public abstract class AbstractDenseNeuron extends Neuron {

    public Connection[] incoming;

    public abstract Pair<Double, Double> compute(double[] input);

}
