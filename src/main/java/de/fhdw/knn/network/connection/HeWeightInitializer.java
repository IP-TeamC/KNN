package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

/**
 * WeightInitializer He
 *
 * @see WeightInitializer#HE
 */
class HeWeightInitializer implements WeightInitializer {

    /**
     * Der WeightInitializer ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link WeightInitializer} erzeugt werden
     */
    HeWeightInitializer() {
    }

    @Override
    public double nextWeight(Random random, Network network, int layer) {
        double fanIn = layer == 0 ? network.inputLayer.neurons.length : network.denseLayers[layer - 1].neurons.length;
        double stddev = Math.sqrt(2.0 / fanIn);
        return random.nextGaussian(0, stddev);
    }

}
