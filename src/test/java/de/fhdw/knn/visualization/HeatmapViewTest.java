package de.fhdw.knn.visualization;

import de.fhdw.knn.TestUtil;
import de.fhdw.knn.network.Network;
import org.jfree.chart.ChartPanel;
import org.junit.jupiter.api.Test;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseWheelEvent;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;

class HeatmapViewTest {

    @Test
    void testHeatmapView() throws InterruptedException, InvocationTargetException {
        SwingUtilities.invokeAndWait(() -> {
            Network network = TestUtil.simpleDummyNetwork();
            HeatmapView view = new HeatmapView();

            view.setNormalizeColors(true);
            view.setShowWeights(true);
            view.setColorSchema(ColorScheme.RED_GREEN);
            view.setThreshold(0.1);

            view.addEpoch("Test-Epoche 1", new HeatmapData(network));
            view.addEpoch("Test-Epoche 2", new HeatmapData(network));

            view.setNormalizeColors(false);
            view.setShowWeights(false);
            view.setColorSchema(ColorScheme.MONOCHROME);
            view.setThreshold(0.5);

            view.showSingleMatrix("Manuelle Test-Gewichtsmatrix", new HeatmapData(network));

            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            SwingUtilities.invokeLater(() -> {
                try {
                    Field tabs = view.getClass().getDeclaredField("tabs");
                    tabs.setAccessible(true);
                    JTabbedPane tabbed = (JTabbedPane) tabs.get(view);
                    int index = tabbed.indexOfTab("Test-Epoche 1");
                    JScrollPane scrollPane = (JScrollPane) tabbed.getComponentAt(index);
                    ChartPanel panel = (ChartPanel) scrollPane.getViewport().getView();
                    panel.restoreAutoBounds();
                    panel.dispatchEvent(new MouseWheelEvent(panel, MouseWheelEvent.MOUSE_WHEEL, System.currentTimeMillis(), 0, 100, 100, 0, false, MouseWheelEvent.WHEEL_UNIT_SCROLL, 3, -1));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });

            view.dispose();
        });
    }

//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapViewCreatesWithoutException() throws InterruptedException, InvocationTargetException {
//        SwingUtilities.invokeAndWait(() -> {
//            HeatmapView view = new HeatmapView();
//            assertNotNull(view);
//            view.dispose();
//        });
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapViewAddEpochDoesNotThrow() throws InterruptedException, InvocationTargetException {
//        SwingUtilities.invokeAndWait(() -> {
//            Network network = TestUtil.simpleDummyNetwork();
//            HeatmapView view = new HeatmapView();
//            assertDoesNotThrow(() -> view.addEpoch("Epoche 1", new HeatmapData(network)));
//            view.dispose();
//        });
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapViewShowSingleMatrixDoesNotThrow() throws InterruptedException, InvocationTargetException {
//        SwingUtilities.invokeAndWait(() -> {
//            Network network = TestUtil.simpleDummyNetwork();
//            HeatmapView view = new HeatmapView();
//            assertDoesNotThrow(() -> view.showSingleMatrix("Test", new HeatmapData(network)));
//            view.dispose();
//        });
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapViewSettersDoNotThrow() throws InterruptedException, InvocationTargetException {
//        SwingUtilities.invokeAndWait(() -> {
//            HeatmapView view = new HeatmapView();
//            assertDoesNotThrow(() -> {
//                view.setNormalizeColors(true);
//                view.setNormalizeColors(false);
//                view.setShowWeights(true);
//                view.setShowWeights(false);
//                view.setColorSchema(ColorScheme.BLUE_RED);
//                view.setColorSchema(ColorScheme.MONOCHROME);
//                view.setThreshold(0.2);
//            });
//            view.dispose();
//        });
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapViewWithNormalizeColors() throws InterruptedException, InvocationTargetException {
//        SwingUtilities.invokeAndWait(() -> {
//            Network network = TestUtil.simpleDummyNetwork();
//            HeatmapView view = new HeatmapView();
//            view.setNormalizeColors(true);
//            assertDoesNotThrow(() -> view.addEpoch("Epoche 1", new HeatmapData(network)));
//            view.dispose();
//        });
//    }
//
//    @Generated("Claude Anthropic AI")
//    @Test
//    void heatmapViewAllColorSchemes() throws InterruptedException, InvocationTargetException {
//        SwingUtilities.invokeAndWait(() -> {
//            Network network = TestUtil.simpleDummyNetwork();
//            for (ColorScheme scheme : ColorScheme.values()) {
//                HeatmapView view = new HeatmapView();
//                view.setColorSchema(scheme);
//                assertDoesNotThrow(() -> view.addEpoch("Epoche 1", new HeatmapData(network)));
//                view.dispose();
//            }
//        });
//    }
}
