package de.fhdw.knn.trainer.stop;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class EarlyStoppingTest {

    @Test
    public void testNoPatience() {
        EarlyStopping earlyStopping = new EarlyStopping(0.1, 0);
        assertFalse(earlyStopping.isFinished(1));
        assertFalse(earlyStopping.isFinished(0.89));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.67));
        assertFalse(earlyStopping.isFinished(0.56));
        assertFalse(earlyStopping.isFinished(0.45));
        assertTrue(earlyStopping.isFinished(0.36));
    }

    @Test
    public void testWithPatience() {
        EarlyStopping earlyStopping = new EarlyStopping(0.1, 5);
        assertFalse(earlyStopping.isFinished(1));
        assertFalse(earlyStopping.isFinished(0.89));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.67));
        assertFalse(earlyStopping.isFinished(0.56));
        assertFalse(earlyStopping.isFinished(0.45));
        assertFalse(earlyStopping.isFinished(0.36));
        assertFalse(earlyStopping.isFinished(0.37));
        assertFalse(earlyStopping.isFinished(0.36));
        assertFalse(earlyStopping.isFinished(0.36));
        assertTrue(earlyStopping.isFinished(0.36));
    }

    @Test
    public void testRecovery() {
        EarlyStopping earlyStopping = new EarlyStopping(0.1, 5);
        assertFalse(earlyStopping.isFinished(1));
        assertFalse(earlyStopping.isFinished(0.89));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.78));
        assertFalse(earlyStopping.isFinished(0.6));
        assertFalse(earlyStopping.isFinished(0.6));
        assertFalse(earlyStopping.isFinished(0.6));
    }

    @Test
    public void testNever() {
        StopFunction never = StopFunction.NEVER;
        for (int i = 0; i < 100; i++) {
            assertFalse(never.isFinished(100));
        }
    }

}
