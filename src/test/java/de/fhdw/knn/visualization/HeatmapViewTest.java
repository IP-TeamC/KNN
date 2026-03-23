package de.fhdw.knn.visualization;

import de.fhdw.knn.util.TestUtil;
import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.util.FakeHeatmapData;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseWheelEvent;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;

import static de.fhdw.knn.util.TestUtil.*;
import static org.junit.jupiter.api.Assertions.*;

class HeatmapViewTest {

    @Test
    void testAddMultipleHeatmap() throws Exception {
        HeatmapView view = new HeatmapView();
        CountDownLatch latch1 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            view.addHeatmap(new HeatmapData(TestUtil.complexDummyNetwork()), "Test-Matrix 1");
            initSleep(100);
            latch1.countDown();
        });

        latch1.await();
        waitForSwing();

        Field tabsField = view.getClass().getDeclaredField("tabs");
        tabsField.setAccessible(true);

        JTabbedPane tabbed = (JTabbedPane) tabsField.get(view);

        assertEquals(1, tabbed.getTabCount());
        assertEquals("Test-Matrix 1", tabbed.getTitleAt(0));

        CountDownLatch latch2 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            view.addHeatmap(new HeatmapData(TestUtil.extremelyComplexDummyNetwork()), "Test-Matrix 2");
            initSleep(100);
            latch2.countDown();
        });

        latch2.await();
        waitForSwing();

        assertEquals(2, tabbed.getTabCount());
        assertEquals("Test-Matrix 2", tabbed.getTitleAt(1));

        initSleep(100);
        view.dispose();
    }

    @Test
    void testAddMultipleHeatmapWithDifferentWeightFilter() throws Exception {
        HeatmapConfig config1 = new HeatmapConfig(0, 0.1, WeightFilter.BOTH, true, true, ColorScheme.MONOCHROME);
        HeatmapView view = new HeatmapView(config1);

        CountDownLatch latch = new CountDownLatch(2);
        SwingUtilities.invokeAndWait(() -> {
            view.addHeatmap(new HeatmapData(TestUtil.complexDummyNetwork()), "Test 1 - WeightFilter.BOTH");
            initSleep(100);
            latch.countDown();
        });

        HeatmapConfig config2 = new HeatmapConfig(0, 0.1, WeightFilter.POSITIVE, true, true, ColorScheme.GREEN_RED);
        view.setConfig(config2);

        SwingUtilities.invokeAndWait(() -> {
            view.addHeatmap(new HeatmapData(TestUtil.extremelyComplexDummyNetwork()), "Test 2 - WeightFilter.POSITIVE");
            initSleep(100);
            latch.countDown();
        });

        HeatmapConfig config3 = new HeatmapConfig(0, 0.1, WeightFilter.NEGATIVE, true, true, ColorScheme.BLUE_RED);
        view.setConfig(config3);

        SwingUtilities.invokeAndWait(() -> {
            view.addHeatmap(new HeatmapData(TestUtil.extremelyComplexDummyNetwork()), "Test 3 - WeightFilter.NEGATIVE");
            initSleep(100);
            latch.countDown();
        });

        latch.await();
        waitForSwing();

        Field tabsField = view.getClass().getDeclaredField("tabs");
        tabsField.setAccessible(true);
        JTabbedPane tabbed = (JTabbedPane) tabsField.get(view);
        assertEquals(3, tabbed.getTabCount());

        JScrollPane scrollPane = (JScrollPane) tabbed.getComponentAt(0);
        ChartPanel panel = (ChartPanel) scrollPane.getViewport().getView();
        assertNotNull(panel.getChart().getXYPlot());

        view.dispose();
    }


    @Test
    void testZoomEvents() throws Exception {
        HeatmapView view = new HeatmapView();

        Field tabsField = view.getClass().getDeclaredField("tabs");
        tabsField.setAccessible(true);
        JTabbedPane tabbed = (JTabbedPane) tabsField.get(view);

        CountDownLatch latch1 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            HeatmapConfig heatmapConfig = new HeatmapConfig(0, 0.1, WeightFilter.BOTH, true, true, ColorScheme.MONOCHROME);
            view.setConfig(heatmapConfig);
            view.addHeatmap(new HeatmapData(TestUtil.complexDummyNetwork()), "Test-Zooming 1");
            initSleep(100);
            latch1.countDown();
        });

        latch1.await();
        waitForSwing();

        // Bounding testen

        JScrollPane scrollPane1 = (JScrollPane) tabbed.getComponentAt(0);
        ChartPanel panel1 = (ChartPanel) scrollPane1.getViewport().getView();

        int panelWidth = panel1.getWidth();
        int panelHeight = panel1.getHeight();

        ArrayList<Pair<Integer, Integer>> coordinates = new ArrayList<>();
        coordinates.add(new Pair<>(105, 100));
        coordinates.add(new Pair<>(panelWidth - 105, 100));
        coordinates.add(new Pair<>(100, panelHeight - 105));
        coordinates.add(new Pair<>(panelWidth - 100, panelHeight - 105));

        for (Pair<Integer, Integer> c : coordinates) {
            for (int j = 0; j < 2; j++) {
                CountDownLatch latch2 = new CountDownLatch(1);
                SwingUtilities.invokeAndWait(() -> {
                    panel1.dispatchEvent(new MouseWheelEvent(panel1, MouseWheelEvent.MOUSE_WHEEL, System.currentTimeMillis(), 0, c.x, c.y, 0, false, MouseWheelEvent.WHEEL_UNIT_SCROLL, 1, -1));
                    initSleep(100);
                    latch2.countDown();
                });
                latch2.await();
            }

            for (int j = 0; j < 3; j++) {
                CountDownLatch latch3 = new CountDownLatch(1);
                SwingUtilities.invokeAndWait(() -> {
                    panel1.dispatchEvent(new MouseWheelEvent(panel1, MouseWheelEvent.MOUSE_WHEEL, System.currentTimeMillis(), 0, c.x, c.y, 0, false, MouseWheelEvent.WHEEL_UNIT_SCROLL, 1, 1));
                    initSleep(100);
                    latch3.countDown();
                });
                latch3.await();
            }
        }

        CountDownLatch latch4 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            HeatmapConfig heatmapConfig = new HeatmapConfig(0, 0.1, WeightFilter.BOTH, true, true, ColorScheme.BLUE_RED);
            view.setConfig(heatmapConfig);
            view.addHeatmap(new HeatmapData(TestUtil.extremelyComplexDummyNetwork()), "Test-Zooming 2");
            initSleep(100);
            latch4.countDown();
        });

        latch4.await();
        waitForSwing();

        // Hereinzoomen bis Zahlen sichtbar

        JScrollPane scrollPane2 = (JScrollPane) tabbed.getComponentAt(1);
        ChartPanel panel2 = (ChartPanel) scrollPane2.getViewport().getView();


        for (int i = 0; i < 25; i++) {
            CountDownLatch latch5 = new CountDownLatch(1);
            SwingUtilities.invokeAndWait(() -> {
                panel2.dispatchEvent(new MouseWheelEvent(panel2, MouseWheelEvent.MOUSE_WHEEL, System.currentTimeMillis(), 0, 600, 200, 0, false, MouseWheelEvent.WHEEL_UNIT_SCROLL, 1, -1));
                initSleep(10);
                latch5.countDown();
            });
            latch5.await();
        }

        for (int i = 0; i < 25; i++) {
            CountDownLatch latch6 = new CountDownLatch(1);
            SwingUtilities.invokeAndWait(() -> {
                panel2.dispatchEvent(new MouseWheelEvent(panel2, MouseWheelEvent.MOUSE_WHEEL, System.currentTimeMillis(), 0, 600, 200, 0, false, MouseWheelEvent.WHEEL_UNIT_SCROLL, 1, 1));
                initSleep(10);
                latch6.countDown();
            });
            latch6.await();
        }

        waitForSwing();

        initSleep(100);
        view.dispose();
    }

    @Test
    void testZeroWeight() throws Exception {
        HeatmapView view = new HeatmapView(
                new HeatmapConfig(0, 0, WeightFilter.BOTH, true, true, ColorScheme.BLUE_RED)
        );

        CountDownLatch latch1 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            Network network = new Network(42, WeightInitializer.ZERO, 2,
                    new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
            DenseNeuron outputNeuron = (DenseNeuron) network.denseLayers[0].neurons[0];
            outputNeuron.incoming[0].weight = 0.0;
            outputNeuron.incoming[1].weight = 0.0;
            outputNeuron.bias = 0.0;

            view.addHeatmap(new HeatmapData(network), "Test-ZeroWeight");
            initSleep(300);
            latch1.countDown();
        });

        latch1.await();
        waitForSwing();

        Field tabsField = view.getClass().getDeclaredField("tabs");
        tabsField.setAccessible(true);

        JTabbedPane tabbed = (JTabbedPane) tabsField.get(view);

        assertEquals(1, tabbed.getTabCount());
        assertEquals("Test-ZeroWeight", tabbed.getTitleAt(0));

        initSleep(300);
        view.dispose();
    }

    /// Testen über Reflection, um 100% Branch-Coverage zu erreichen.
    @Generated("ChatGPT")
    @Test
    void testRendererPaintScaleAndBoundsViaReflection() throws Exception {
        Method method = HeatmapView.class.getDeclaredMethod("getXyBlockRenderer",
                double.class, double.class, double.class, boolean.class, WeightFilter.class, ColorScheme.class);
        method.setAccessible(true);

        // Renderer erzeugen
        XYBlockRenderer renderer = (XYBlockRenderer) method.invoke(
                null, -1.0, 1.0, 0.1, true, WeightFilter.BOTH, ColorScheme.GREEN_RED);
        assertNotNull(renderer);

        // PaintScale aus Renderer extrahieren
        var paintScaleField = XYBlockRenderer.class.getDeclaredField("paintScale");
        paintScaleField.setAccessible(true);
        Object paintScaleObj = paintScaleField.get(renderer);
        assertNotNull(paintScaleObj);

        // getLowerBound() / getUpperBound() testen
        Method getLower = paintScaleObj.getClass().getDeclaredMethod("getLowerBound");
        Method getUpper = paintScaleObj.getClass().getDeclaredMethod("getUpperBound");
        double lower = (double) getLower.invoke(paintScaleObj);
        double upper = (double) getUpper.invoke(paintScaleObj);
        assertEquals(-1.0, lower);
        assertEquals(1.0, upper);

        // getPaint() testen: alle relevanten Fälle
        Method getPaint = paintScaleObj.getClass().getDeclaredMethod("getPaint", double.class);

        Paint pNaN = (Paint) getPaint.invoke(paintScaleObj, Double.NaN); // NaN → Grauer Block
        assertNotNull(pNaN);

        Paint pUnderThreshold = (Paint) getPaint.invoke(paintScaleObj, 0.05); // Unter Threshold → weiß
        assertNotNull(pUnderThreshold);

        Paint pPositive = (Paint) getPaint.invoke(paintScaleObj, 0.8); // Positiv interpoliert
        assertNotNull(pPositive);

        Paint pNegative = (Paint) getPaint.invoke(paintScaleObj, -0.8); // Negativ interpoliert
        assertNotNull(pNegative);

        // WeightFilter.POSITIVE → negative Werte werden weiß
        XYBlockRenderer rendererPos = (XYBlockRenderer) method.invoke(
                null, -1.0, 1.0, 0.1, true, WeightFilter.POSITIVE, ColorScheme.GREEN_RED);
        Object psPos = paintScaleField.get(rendererPos);
        Method getPaintPos = psPos.getClass().getDeclaredMethod("getPaint", double.class);
        assertEquals(Color.WHITE, (Paint) getPaintPos.invoke(psPos, -0.8)); // NEGATIVE → gefiltert → weiß
        assertNotEquals(Color.WHITE, (Paint) getPaintPos.invoke(psPos, 0.8)); // POSITIVE → eingefärbt

        // WeightFilter.NEGATIVE → positive Werte werden weiß
        XYBlockRenderer rendererNeg = (XYBlockRenderer) method.invoke(
                null, -1.0, 1.0, 0.1, true, WeightFilter.NEGATIVE, ColorScheme.GREEN_RED);
        Object psNeg = paintScaleField.get(rendererNeg);
        Method getPaintNeg = psNeg.getClass().getDeclaredMethod("getPaint", double.class);
        assertEquals(Color.WHITE, (Paint) getPaintNeg.invoke(psNeg, 0.8));  // POSITIVE → gefiltert → weiß
        assertNotEquals(Color.WHITE, (Paint) getPaintNeg.invoke(psNeg, -0.8)); // NEGATIVE → eingefärbt
    }

    @Generated("Claude AI")
    @Test
    void testInvalidXYZDataset() throws Exception {
        HeatmapView view = new HeatmapView(
                new HeatmapConfig(1, 0.0, WeightFilter.BOTH, true, true, ColorScheme.GREEN_RED)
        );

        CountDownLatch latch1 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            view.addHeatmap(new FakeHeatmapData(simpleDummyNetwork()), "Test-NonXYZBranch");
            initSleep(100);
            latch1.countDown();
        });

        latch1.await();
        waitForSwing();

        JTabbedPane tabs = (JTabbedPane) view.getContentPane();
        JScrollPane scroll = (JScrollPane) tabs.getComponentAt(tabs.getSelectedIndex());
        ChartPanel panel = (ChartPanel) scroll.getViewport().getView();

        BufferedImage img = new BufferedImage(100, 100, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = img.createGraphics();
        CountDownLatch latch2 = new CountDownLatch(1);
        SwingUtilities.invokeAndWait(() -> {
            panel.paint(g2);
            initSleep(100);
            latch2.countDown();
        });

        latch2.await();
        waitForSwing();

        initSleep(100);
        view.dispose();
    }
}
