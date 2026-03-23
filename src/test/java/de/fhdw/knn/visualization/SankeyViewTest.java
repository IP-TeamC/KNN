package de.fhdw.knn.visualization;

import de.fhdw.knn.util.TestUtil;
import de.fhdw.knn.network.Network;
import javafx.application.Platform;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static de.fhdw.knn.util.TestUtil.waitForJavaFX;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SankeyViewTest {

    @Test
    void testSimpleSankeyView() throws Exception {
        TestUtil.startPlatform();
        CountDownLatch latch = new CountDownLatch(1);

        assertDoesNotThrow(() -> Platform.runLater(() -> {
            try {
                Network network1 = TestUtil.simpleDummyNetwork();
                SankeyView view = new SankeyView(new SankeyData(network1), "Test-Epoche 1");

                Network network2 = TestUtil.simpleDummyNetwork();
                view.update(new SankeyData(network2), "Test-Epoche 2");
            } finally {
                latch.countDown();
            }
        }));

        assertTrue(latch.await(5, TimeUnit.SECONDS));
        waitForJavaFX();
    }

    @Test
    void testComplexSankeyView() throws Exception {
        TestUtil.startPlatform();

        CountDownLatch latch = new CountDownLatch(1);

        assertDoesNotThrow(() -> Platform.runLater(() -> {
            try {
                Network network1 = TestUtil.complexDummyNetwork();
                SankeyView view = new SankeyView(new SankeyData(network1), "Test-Epoche 1");

                Network network2 = TestUtil.complexDummyNetwork();
                view.update(new SankeyData(network2), "Test-Epoche 2");
            } finally {
                latch.countDown();
            }
        }));

        assertTrue(latch.await(5, TimeUnit.SECONDS));
        waitForJavaFX();
    }

    @Test
    void testChangeConfig() throws Exception {
        TestUtil.startPlatform();

        CountDownLatch latch = new CountDownLatch(1);

        assertDoesNotThrow(() -> Platform.runLater(() -> {
            try {
                Network network1 = TestUtil.complexDummyNetwork();
                SankeyView view = new SankeyView(new SankeyData(network1), "Test-Epoche 1");
                view.setThreshold(0.2);
                view.setWeightFilter(WeightFilter.POSITIVE);
                Network network2 = TestUtil.complexDummyNetwork();
                view.update(new SankeyData(network2), "Test-Epoche 2");
            } finally {
                latch.countDown();
            }
        }));

        assertTrue(latch.await(5, TimeUnit.SECONDS));
        waitForJavaFX();
    }

    @Test
    void testExtremelyComplexSankeyView() throws Exception {
        TestUtil.startPlatform();
        CountDownLatch latch = new CountDownLatch(1);

        assertDoesNotThrow(() -> Platform.runLater(() -> {
            try {
                Network network1 = TestUtil.extremelyComplexDummyNetwork();
                SankeyView view = new SankeyView(new SankeyData(network1), "Test-Epoche 1");

                Network network2 = TestUtil.extremelyComplexDummyNetwork();
                view.update(new SankeyData(network2), "Test-Epoche 2");
            } finally {
                latch.countDown();
            }
        }));

        assertTrue(latch.await(5, TimeUnit.SECONDS));
        waitForJavaFX();
    }
}