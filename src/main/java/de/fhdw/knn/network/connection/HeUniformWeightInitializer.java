package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

/**
 * WeightInitializer He (Uniform)
 *
 * @see WeightInitializer#HE_UNIFORM
 */
class HeUniformWeightInitializer implements WeightInitializer {

    /**
     * Der WeightInitializer ist zustandslos und es sollten keine weiteren Instanzen außerhalb von {@link WeightInitializer} erzeugt werden
     */
    HeUniformWeightInitializer() {
    }

    @Override
    public double nextWeight(Random random, Network network, int layer) {
        double fanIn = layer == 0 ? network.inputLayer.neurons.length : network.denseLayers[layer - 1].neurons.length;
        double bound = Math.sqrt(6.0 / fanIn);
        double absWeight = random.nextDouble(bound);
        return random.nextBoolean() ? absWeight : -absWeight;
    }

}
