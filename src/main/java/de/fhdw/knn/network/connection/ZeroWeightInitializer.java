package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

/**
 * @see WeightInitializer#ZERO
 */
class ZeroWeightInitializer implements WeightInitializer {

    @Override
    public double nextWeight(Random random, Network network, int layer) {
        return 0;
    }

}
