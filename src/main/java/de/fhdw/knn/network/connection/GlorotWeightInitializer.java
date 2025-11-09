package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;

import java.util.Random;

public class GlorotWeightInitializer implements WeightInitializer {

    @Override
    public double nextWeight(Random random, Network network, int layer) {
        double fanIn = layer == 0 ? network.inputLayer.neurons.length : network.denseLayers[layer - 1].neurons.length;
        double fanOut = network.denseLayers[layer].neurons.length;
        double stddev = Math.sqrt(2.0 / (fanIn + fanOut));
        return random.nextGaussian(0, stddev);
    }

}
