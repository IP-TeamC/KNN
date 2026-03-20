package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.util.TestUtil;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import javax.swing.*;

import static de.fhdw.knn.util.TestUtil.waitForSwing;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

class ViewManagerTest {

    @Test
    void viewManagerWithHeatmapDoesNotThrow() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            Network network = TestUtil.simpleDummyNetwork();
            ViewManager vm = new ViewManager(3, 0, true, true, ColorScheme.RED_GREEN, network);
            assertDoesNotThrow(() -> vm.nextEpoch(1, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(2, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(3, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(4, network, 4));
            assertDoesNotThrow(() -> vm.earlyStop(0, network));
        });
        waitForSwing();
    }

    @Test
    void viewManagerWithSankeyDoesNotThrow() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            Network network = TestUtil.simpleDummyNetwork();
            ViewManager vm = new ViewManager(0, 3, true, true, ColorScheme.RED_GREEN, network);
            assertDoesNotThrow(() -> vm.nextEpoch(1, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(2, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(3, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(4, network, 4));
            assertDoesNotThrow(() -> vm.earlyStop(0, network));
        });
        waitForSwing();
    }

    @Generated("Claude AI")
    @Test
    void viewManagerDeactivatedDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        assertDoesNotThrow(() -> new ViewManager(0, 0, true, false, ColorScheme.RED_GREEN, network));
    }

    @Generated("Claude AI")
    @Test
    void viewManagerNextEpochDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(1, 1, true, false, ColorScheme.RED_GREEN, network);
        assertDoesNotThrow(() -> vm.nextEpoch(1, network, 10));
        assertDoesNotThrow(() -> vm.nextEpoch(5, network, 10));
        assertDoesNotThrow(() -> vm.nextEpoch(10, network, 10));
    }

    @Generated("Claude AI")
    @Test
    void viewManagerEarlyStopDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(1, 1, true, false, ColorScheme.RED_GREEN, network);
        assertDoesNotThrow(() -> vm.earlyStop(5, network));
    }
}
