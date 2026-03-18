package de.fhdw.knn.data;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class PairTest {

    @Test
    public void testPairConstructorAndParameters() {
        Integer x = 42;
        String y = "test";
        Pair<Integer, String> pair = new Pair<>(x, y);

        assertEquals(x, pair.x);
        assertEquals(y, pair.y);
    }
}
