package de.fhdw.knn.config.visualization;

import de.fhdw.knn.util.TestUtil;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConfVisualizationTest {

    @Test
    void testConfVisualizationCreateWithHeatmapAndSankeyInterval() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmapInterval = 1;
        conf.sankeyInterval = 1;
        conf.colorScheme = "RED_GREEN";

        assertNotNull(conf.create(TestUtil.simpleDummyNetwork()));
        assertTrue(conf.isEnabled());
    }

    @Test
    void testConfVisualizationCreateWithSankeyInterval() {
        ConfVisualization conf = new ConfVisualization();
        conf.sankeyInterval = 1;
        conf.colorScheme = "MONOCHROME";

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
    void testConfVisualizationCreateWithInvalidColorScheme() {
        ConfVisualization conf = new ConfVisualization();
        conf.heatmapInterval = 1;
        conf.colorScheme = "INVALID";

        assertThrows(Exception.class, () -> conf.create(TestUtil.simpleDummyNetwork()));
    }
}