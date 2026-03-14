package de.fhdw.knn;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;

public class TestUtil {

    public static Network simpleDummyNetwork() {
        return new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1));
    }

}
