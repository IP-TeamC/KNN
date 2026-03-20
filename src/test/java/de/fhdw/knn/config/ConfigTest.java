package de.fhdw.knn.config;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.util.TestUtil;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConfigTest {

    @Test
    void testReadAndExecuteMinimal() {
        Network network = TestUtil.simpleDummyNetwork();
        network.export("models/test_conf.knn");

        Config config = Config.read("conf/test_conf_import.toml");
        assertDoesNotThrow(config::execute);
    }

    @Test
    void testExecuteWithTrainerAndVisualization() {
        Config config = Config.read("conf/test_conf_trainer_visualization.toml");
        assertDoesNotThrow(config::execute);
    }

    @Test
    void testReadAndExecuteWithVisualizationOnly() {
        Config config = Config.read("conf/test_conf_no_trainer_visualization.toml");
        assertDoesNotThrow(config::execute);
    }

    @Test
    void testExecuteTrainerVisualizationDisabled() {
        Config config = Config.read("conf/test_conf_visualization_disabled_with_trainer.toml");
        assertDoesNotThrow(config::execute);
    }

    @Test
    void testExecuteNoTrainerVisualizationDisabled() {
        Config config = Config.read("conf/test_conf_visualization_disabled_without_trainer.toml");
        assertDoesNotThrow(config::execute);
    }

    @Test
    void testExecuteTrainerWithoutVisualization() {
        Config config = Config.read("conf/test_conf_trainer_no_visualization.toml");
        assertDoesNotThrow(config::execute);
    }

    @Test
    void testGetStaticField() {
        LossFunction mse = Config.getStaticField(LossFunction.class, "MEAN_SQUARED_ERROR");
        assertNotNull(mse);
    }

    @Test
    void testGetStaticFieldThrows() {
        assertThrows(Exception.class, () -> Config.getStaticField(LossFunction.class, "MÖGE_DIE_MACHT_MIT_DIR_SEIN"));
    }
}