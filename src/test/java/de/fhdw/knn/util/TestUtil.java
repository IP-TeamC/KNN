package de.fhdw.knn.util;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;

import javax.swing.*;
import java.lang.reflect.InvocationTargetException;

public class TestUtil {

    public static Network simpleDummyNetwork() {
        return new Network(42, WeightInitializer.GLOROT_UNIFORM, 1,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 1));
    }

    public static Network complexDummyNetwork() {
        return new Network(42, WeightInitializer.GLOROT_UNIFORM, 2,
                DenseLayer.createLayers(null, ActivationFunction.LINEAR, 10, 5, 2));
    }

    public static Network extremelyComplexDummyNetwork() {
        return new Network(42, WeightInitializer.GLOROT_UNIFORM, 5,
                DenseLayer.createLayers(null, ActivationFunction.RELU, 100, 50, 100, 10));
    }

    public static void waitForSwing() {
        try {
            SwingUtilities.invokeAndWait(() -> {});
        } catch (InvocationTargetException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}

