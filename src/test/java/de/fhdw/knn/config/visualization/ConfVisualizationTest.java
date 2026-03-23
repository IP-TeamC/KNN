package de.fhdw.knn.config.visualization;

import de.fhdw.knn.util.TestUtil;
import de.fhdw.knn.visualization.ColorScheme;
import de.fhdw.knn.visualization.WeightFilter;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

class ConfVisualizationTest {

    @Test
    void testConfVisualizationCreateWithHeatmapAndSankeyInterval() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmap.interval = 1;
        conf.heatmap.colorScheme = "GREEN_RED";
        conf.sankey.interval = 1;

        assertNotNull(conf.create(TestUtil.simpleDummyNetwork()));
        assertTrue(conf.isEnabled());
    }

    @Test
    void testConfVisualizationCreateWithOnlyHeatmapEnabled() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmap.interval = 1;
        conf.heatmap.colorScheme = "GREEN_RED";

        assertNotNull(conf.create(TestUtil.simpleDummyNetwork()));
        assertTrue(conf.isEnabled());
    }

    @Test
    void testConfVisualizationCreateWithOnlySankeyEnabled() {
        ConfVisualization conf = new ConfVisualization();
        conf.sankey.interval = 1;

        assertNotNull(conf.create(TestUtil.simpleDummyNetwork()));
        assertTrue(conf.isEnabled());
    }

    @Test
    void testConfVisualizationCreateReturnsNullWhenDisabled() {
        ConfVisualization conf = new ConfVisualization();
        assertNull(conf.create(TestUtil.simpleDummyNetwork()));
        assertFalse(conf.isEnabled());
    }

    @Test
    void testConfVisualizationCreateWithEndOnlyIntervals() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmap.interval = -1;
        conf.sankey.interval = -1;

        assertNotNull(conf.create(TestUtil.simpleDummyNetwork()));
        assertTrue(conf.isEnabled());
    }

    @Test
    void testConfVisualizationCreateWithInvalidColorScheme() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmap.interval = 1;
        conf.heatmap.colorScheme = "INVALID";

        assertThrows(Exception.class, () -> conf.create(TestUtil.simpleDummyNetwork()));
    }

    @Generated("Claude AI")
    @Test
    void testCreateDoesNotThrowOnValidColorSchemes() {
        for (ColorScheme scheme : ColorScheme.values()) {
            ConfVisualization conf = new ConfVisualization();
            conf.heatmap.interval = 1;
            conf.heatmap.colorScheme = scheme.name();

            assertDoesNotThrow(() -> conf.create(TestUtil.simpleDummyNetwork()),
                    "colorScheme '" + scheme + "' sollte gültig sein");
        }
    }

    @Generated("Claude AI")
    @Test
    void testCreateThrowsOnInvalidHeatmapWeightFilter() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmap.interval = 1;
        conf.heatmap.weightFilter = "INVALID";

        assertThrows(IllegalArgumentException.class,
                () -> conf.create(TestUtil.simpleDummyNetwork()));
    }

    @Generated("Claude AI")
    @Test
    void testCreateThrowsOnInvalidSankeyWeightFilter() {
        ConfVisualization conf = new ConfVisualization();
        conf.sankey.interval = 1;
        conf.sankey.weightFilter = "INVALID";

        assertThrows(IllegalArgumentException.class,
                () -> conf.create(TestUtil.simpleDummyNetwork()));
    }

    @Generated("Claude AI")
    @Test
    void testCreateDoesNotThrowOnValidWeightFilters() {
        for (WeightFilter filter : WeightFilter.values()) {
            ConfVisualization conf = new ConfVisualization();
            conf.heatmap.interval = 1;
            conf.heatmap.weightFilter = filter.name();
            conf.sankey.interval = 1;
            conf.sankey.weightFilter = filter.name();

            assertDoesNotThrow(() -> conf.create(TestUtil.simpleDummyNetwork()),
                    "weightFilter '" + filter + "' sollte gültig sein");
        }
    }

    @Generated("Claude AI")
    @Test
    void testDefaultValuesHeatmap() {
        ConfHeatmap heatmap = new ConfHeatmap();
        assertEquals(0, heatmap.interval);
        assertEquals(0.1, heatmap.threshold);
        assertEquals("BOTH", heatmap.weightFilter);
        assertEquals("GREEN_RED", heatmap.colorScheme);
        assertTrue(heatmap.showWeights);
        assertTrue(heatmap.normalizeColors);
    }

    @Generated("Claude AI")
    @Test
    void testDefaultValuesSankey() {
        ConfSankey sankey = new ConfSankey();
        assertEquals(0, sankey.interval);
        assertEquals(0.1, sankey.threshold);
        assertEquals("BOTH", sankey.weightFilter);
    }

    @Generated("Claude AI")
    @Test
    void testDefaultConfVisualizationIsDisabled() {
        ConfVisualization conf = new ConfVisualization();
        assertFalse(conf.isEnabled());
        assertNull(conf.create(TestUtil.simpleDummyNetwork()));
    }
}