package de.fhdw.knn.util;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import javafx.application.Platform;

import javax.swing.*;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.CountDownLatch;

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

    public static void initSleep(long time) {
        try {
            Thread.sleep(time);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void waitForSwing() {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            SwingUtilities.invokeAndWait(latch::countDown);
            latch.await();
        } catch (InvocationTargetException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void waitForJavaFX() {
        try {
            CountDownLatch latch = new CountDownLatch(1);
            Platform.runLater(latch::countDown);
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void startPlatform() {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {}
    }
}

