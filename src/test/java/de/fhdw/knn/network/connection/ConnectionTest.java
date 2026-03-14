package de.fhdw.knn.network.connection;

import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

public class ConnectionTest {

    // Die Klasse Connection wurde bereits in anderen Testfällen ausreichend getestet,
    // hier wurden nur einige ergänzende Testfälle zusätzlich erzeugt.

    @Generated("GitHub Copilot")
    @Test
    public void testConstructorWithWeightSetsGuardTrue() {
        Connection connection = new Connection(1.5);

        assertTrue(connection.guard);
        assertEquals(1.5, connection.weight, 1e-12);
        assertNull(connection.inputNeuron);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testConstructorWithWeightAndGuardSetsBothValues() {
        Connection guardedOff = new Connection(2.5, false);
        Connection guardedOn = new Connection(-3.0, true);

        assertFalse(guardedOff.guard);
        assertEquals(2.5, guardedOff.weight, 1e-12);

        assertTrue(guardedOn.guard);
        assertEquals(-3.0, guardedOn.weight, 1e-12);
    }
}

