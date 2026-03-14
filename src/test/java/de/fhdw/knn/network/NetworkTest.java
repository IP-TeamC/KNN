package de.fhdw.knn.network;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class NetworkTest {

    // Testet nur Code, der nicht bereits zuvor in anderen Tests des Packages getestet wurde

    @Test
    public void testNonOverridingConnector() {
        InputLayer inputLayer = new InputLayer(3);
        DenseLayer denseLayer1 = new DenseLayer(2).withActivationFunction(ActivationFunction.RELU);
        DenseLayer denseLayer2 = new DenseLayer(5).withActivationFunction(ActivationFunction.TANH);
        denseLayer1.neurons[0].incoming = new Connection[]{
                new Connection(3),
                new Connection(1),
                new Connection(5)
        };
        denseLayer2.neurons[3].incoming = new Connection[]{
                new Connection(4),
                new Connection(2),
        };
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM,
                inputLayer, denseLayer1, denseLayer2);
        network.predictSingles(new double[][]{
                new double[]{1, 2, 3},
                new double[]{4, 2, 0}
        });

        assertEquals(3, network.denseLayers[0].neurons[0].incoming[0].weight);
        assertEquals(1, network.denseLayers[0].neurons[0].incoming[1].weight);
        assertEquals(5, network.denseLayers[0].neurons[0].incoming[2].weight);
        assertEquals(4, network.denseLayers[1].neurons[3].incoming[0].weight);
        assertEquals(2, network.denseLayers[1].neurons[3].incoming[1].weight);
    }

}
