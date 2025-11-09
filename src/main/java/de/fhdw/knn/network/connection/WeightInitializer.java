package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

public interface WeightInitializer {

    WeightInitializer GLOROT = new GlorotWeightInitializer();
    WeightInitializer HE = new HeWeightInitializer();
    WeightInitializer ZERO = new ZeroWeightInitializer();

    double nextWeight(Random random, Network network, int layer);

}