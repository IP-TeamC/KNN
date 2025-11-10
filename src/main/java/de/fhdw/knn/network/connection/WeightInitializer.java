package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

public interface WeightInitializer {

    WeightInitializer GLOROT = new GlorotWeightInitializer();
    WeightInitializer GLOROT_UNIFORM = new GlorotUniformWeightInitializer();
    WeightInitializer HE = new HeWeightInitializer();
    WeightInitializer HE_UNIFORM = new HeUniformWeightInitializer();
    WeightInitializer ZERO = new ZeroWeightInitializer();

    double nextWeight(Random random, Network network, int layer);

}