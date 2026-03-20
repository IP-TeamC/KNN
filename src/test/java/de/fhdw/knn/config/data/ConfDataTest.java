package de.fhdw.knn.config.data;

import de.fhdw.knn.data.TrainTestSplit;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class ConfDataTest {

    @Test
    void testConfDataCreate() throws IOException {
        ConfData conf = new ConfData();
        conf.file = "data/test_conf.csv";
        conf.inputStart = 0;
        conf.inputSize = 3;
        conf.outputStart = 3;
        conf.outputSize = 2;
        conf.skip = 0;
        conf.seed = 42;
        conf.testShare = 0.2;

        TrainTestSplit split = conf.create();

        assertNotNull(split);
        assertNotNull(split.train);
        assertNotNull(split.test);

        // 10 Zeilen, 20 % Test → 2 Test, 8 Train
        assertEquals(8, split.train.size);
        assertEquals(2, split.test.size);
    }
}