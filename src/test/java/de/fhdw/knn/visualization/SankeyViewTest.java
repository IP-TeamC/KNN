package de.fhdw.knn.visualization;

import de.fhdw.knn.TestUtil;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import javafx.application.Platform;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

class SankeyViewTest {

    @Test
    void testSankeyView() throws InterruptedException {
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException ignored) {}

        CountDownLatch latch = new CountDownLatch(1);
        Platform.runLater(() -> {
            Network network1 = TestUtil.simpleDummyNetwork();

            SankeyView view = new SankeyView();
            assertDoesNotThrow(() -> view.show(network1));

            InputLayer inputLayer = new InputLayer(10);
            DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 5, 5, 2);
            Network network2 = new Network(42, WeightInitializer.HE, inputLayer, denseLayers);
            assertDoesNotThrow(() -> view.update(network2, "Test-Epoche"));

            latch.countDown();
        });

        latch.await(5, TimeUnit.SECONDS);
    }
}