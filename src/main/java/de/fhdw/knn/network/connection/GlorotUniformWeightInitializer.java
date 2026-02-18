package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

class GlorotUniformWeightInitializer implements WeightInitializer {

    @Override
    public double nextWeight(Random random, Network network, int layer) {
        double fanIn = layer == 0 ? network.inputLayer.neurons.length : network.denseLayers[layer - 1].neurons.length;
        double fanOut = network.denseLayers[layer].neurons.length;
        double bound = Math.sqrt(6.0 / (fanIn + fanOut));
        double absWeight = random.nextDouble(bound);
        return random.nextBoolean() ? absWeight : -absWeight;
    }

}
