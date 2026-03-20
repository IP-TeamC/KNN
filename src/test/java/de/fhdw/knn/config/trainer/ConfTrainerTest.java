package de.fhdw.knn.config.trainer;

import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.util.TestUtil;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConfTrainerTest {

    @Test
    void testConfNetworkCreateWithLogLoss() {
        ConfTrainer conf = new ConfTrainer();
        conf.lossFunction = "MEAN_SQUARED_ERROR";
        conf.maxEpochs = 10;
        conf.shuffleEpoch = true;
        conf.batchSize = 1;
        conf.learningRate = 0.01;
        conf.logLoss = true;
        conf.exportFile = "models/test_conf.knn";

        Trainer trainer = conf.create(TestUtil.simpleDummyNetwork());
        assertNotNull(trainer);
    }

    @Test
    void testConfNetworkCreateWithoutLogLoss() {
        ConfTrainer conf = new ConfTrainer();
        conf.lossFunction = "CROSS_ENTROPY_LOSS";
        conf.maxEpochs = 10;
        conf.shuffleEpoch = true;
        conf.batchSize = 1;
        conf.learningRate = 0.01;
        conf.logLoss = false;

        Trainer trainer = conf.create(TestUtil.simpleDummyNetwork());
        assertNotNull(trainer);
    }

    @Test
    void testExportWithFile() {
        ConfTrainer conf = new ConfTrainer();
        conf.exportFile = "models/test_conf.knn";

        assertDoesNotThrow(() -> conf.export(TestUtil.simpleDummyNetwork()));
    }

    @Test
    void testExportWithoutFile() {
        ConfTrainer conf = new ConfTrainer();
        conf.exportFile = null;

        assertDoesNotThrow(() -> conf.export(TestUtil.simpleDummyNetwork()));
    }
}