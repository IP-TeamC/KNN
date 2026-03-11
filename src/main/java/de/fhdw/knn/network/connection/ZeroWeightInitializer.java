package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

/**
 * 0-WeightInitializer
 *
 * @see WeightInitializer#ZERO
 */
class ZeroWeightInitializer implements WeightInitializer {

    /**
     * Der WeightInitializer ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link WeightInitializer} erzeugt werden
     */
    ZeroWeightInitializer() {
    }

    @Override
    public double nextWeight(Random random, Network network, int layer) {
        return 0;
    }

}
