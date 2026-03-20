package de.fhdw.knn.config.trainer;

import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConfEarlyStoppingTest {

    @Test
    void testConfEarlyStoppingCreate() {
        ConfEarlyStopping conf = new ConfEarlyStopping();
        conf.minDelta = 0.001;
        conf.patience = 5;

        StopFunction stopFunction = conf.create();
        assertNotNull(stopFunction);
        assertInstanceOf(EarlyStopping.class, stopFunction);
    }

}