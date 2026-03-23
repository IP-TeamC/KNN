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
            ViewManager vm = new ViewManager(HeatmapConfig.withDefaults(3), null, network);
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
            ViewManager vm = new ViewManager(null, SankeyConfig.withDefaults(3), network);
            assertDoesNotThrow(() -> vm.nextEpoch(1, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(2, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(3, network, 4));
            assertDoesNotThrow(() -> vm.nextEpoch(4, network, 4));
            assertDoesNotThrow(() -> vm.earlyStop(0, network));
        });
        waitForSwing();
    }

    @Test
    void viewManagerInitNetworkNullDoesNotThrow() throws Exception {
        SwingUtilities.invokeAndWait(() ->
                assertDoesNotThrow(() -> new ViewManager(
                        HeatmapConfig.withDefaults(1),
                        SankeyConfig.withDefaults(1),
                        null
                ))
        );
        waitForSwing();
    }

    @Test
    void viewManagerEarlyStopCreatesSankeyIfNull() throws Exception {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(null, SankeyConfig.withDefaults(-1), network);
        SwingUtilities.invokeAndWait(() -> assertDoesNotThrow(() -> vm.earlyStop(5, network)));
        waitForSwing();
    }

    @Test
    void viewManagerEarlyStopUpdatesSankeyIfExists() throws Exception {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(null, SankeyConfig.withDefaults(1), network);
        SwingUtilities.invokeAndWait(() -> assertDoesNotThrow(() -> vm.earlyStop(5, network)));
        waitForSwing();
    }

    @Test
    void viewManagerEarlyStopHeatmapNull() throws Exception {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(null, null, network);
        SwingUtilities.invokeAndWait(() -> assertDoesNotThrow(() -> vm.earlyStop(5, network)));
        waitForSwing();
    }

    @Test
    void viewManagerEarlyStopHeatmapNotNull() throws Exception {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(HeatmapConfig.withDefaults(1), null, network);
        SwingUtilities.invokeAndWait(() -> assertDoesNotThrow(() -> vm.earlyStop(5, network)));
        waitForSwing();
    }

    @Test
    void viewManagerEarlyStopHeatmapIntervalZero() throws Exception {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(HeatmapConfig.withDefaults(0), null, network);
        SwingUtilities.invokeAndWait(() -> assertDoesNotThrow(() -> vm.earlyStop(5, network)));
        waitForSwing();
    }

    @Test
    void viewManagerEarlyStopSankeyIntervalZero() throws Exception {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(null, SankeyConfig.withDefaults(0), network);
        SwingUtilities.invokeAndWait(() -> assertDoesNotThrow(() -> vm.earlyStop(5, network)));
        waitForSwing();
    }

    @Generated("Claude AI")
    @Test
    void viewManagerDeactivatedDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        assertDoesNotThrow(() -> new ViewManager(null, null, network));
    }

    @Generated("Claude AI")
    @Test
    void viewManagerBothEnabledDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        assertDoesNotThrow(() -> new ViewManager(HeatmapConfig.withDefaults(1), SankeyConfig.withDefaults(1), network));
    }

    @Generated("Claude AI")
    @Test
    void viewManagerNextEpochDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(HeatmapConfig.withDefaults(1), SankeyConfig.withDefaults(1), network);
        assertDoesNotThrow(() -> vm.nextEpoch(1, network, 10));
        assertDoesNotThrow(() -> vm.nextEpoch(5, network, 10));
        assertDoesNotThrow(() -> vm.nextEpoch(10, network, 10));
    }

    @Generated("Claude AI")
    @Test
    void viewManagerEarlyStopDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(HeatmapConfig.withDefaults(1), SankeyConfig.withDefaults(1), network);
        assertDoesNotThrow(() -> vm.earlyStop(5, network));
    }

    @Generated("Claude AI")
    @Test
    void viewManagerEndOnlyIntervalsDoesNotThrow() {
        Network network = TestUtil.simpleDummyNetwork();
        ViewManager vm = new ViewManager(HeatmapConfig.withDefaults(-1), SankeyConfig.withDefaults(-1), network);
        assertDoesNotThrow(() -> vm.nextEpoch(1, network, 10));
        assertDoesNotThrow(() -> vm.nextEpoch(10, network, 10));
        assertDoesNotThrow(() -> vm.earlyStop(5, network));
    }
}
