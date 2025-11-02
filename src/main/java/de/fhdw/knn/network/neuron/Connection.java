package de.fhdw.knn.network.neuron;

import lombok.ToString;

import java.util.Random;

@ToString
public class Connection {

    public static final Random RANDOM = new Random();

    @ToString.Exclude
    public boolean guard = true;
    public double weight = RANDOM.nextGaussian();
    @ToString.Exclude
    public Neuron inputNeuron;

}
