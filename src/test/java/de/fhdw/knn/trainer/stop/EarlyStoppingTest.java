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

    @Test
    public void testPatienceOne() {
        // patience=1 → stoppt beim ersten aufeinanderfolgenden Nicht-Fortschritt
        EarlyStopping earlyStopping = new EarlyStopping(0.1, 1);
        assertFalse(earlyStopping.isFinished(1.0));   // großer Fortschritt
        assertFalse(earlyStopping.isFinished(0.85));  // Fortschritt 0.15 > 0.1 → patience wird zurückgesetzt
        assertTrue(earlyStopping.isFinished(0.85));   // kein Fortschritt → epochsWaited=1 >= patience=1 → stop
    }

    @Test
    public void testImprovementExactlyAtMinDelta() {
        // Verbesserung genau gleich minDelta gilt als ausreichend: kein Patience-Inkrement
        // Bedingung: previousLoss - loss < minDelta → bei Gleichheit ist dies false → Fortschritt
        EarlyStopping earlyStopping = new EarlyStopping(0.1, 3);
        assertFalse(earlyStopping.isFinished(1.0));  // Fortschritt
        assertFalse(earlyStopping.isFinished(0.9));  // Verbesserung = 0.1 = minDelta → gilt als Fortschritt
        assertFalse(earlyStopping.isFinished(0.8));  // Verbesserung = 0.1 = minDelta → gilt als Fortschritt
        assertFalse(earlyStopping.isFinished(0.7));  // Verbesserung = 0.1 = minDelta → gilt als Fortschritt
        assertFalse(earlyStopping.isFinished(0.6));  // Verbesserung = 0.1 = minDelta → gilt als Fortschritt
    }

    @Test
    public void testStopsAfterExactlyPatience() {
        // Stoppt nach genau patience aufeinanderfolgenden Epochen ohne ausreichende Verbesserung
        EarlyStopping earlyStopping = new EarlyStopping(0.1, 3);
        assertFalse(earlyStopping.isFinished(1.0));  // großer Fortschritt
        assertFalse(earlyStopping.isFinished(0.5));  // großer Fortschritt
        assertFalse(earlyStopping.isFinished(0.5));  // kein Fortschritt, epochsWaited=1
        assertFalse(earlyStopping.isFinished(0.5));  // kein Fortschritt, epochsWaited=2
        assertTrue(earlyStopping.isFinished(0.5));   // kein Fortschritt, epochsWaited=3 >= patience=3 → stop
    }
}
