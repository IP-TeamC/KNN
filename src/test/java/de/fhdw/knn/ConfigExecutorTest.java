package de.fhdw.knn;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

public class ConfigExecutorTest {

    @Test
    public void testRun() {
        ConfigExecutor.main(new String[]{"test_bq_simple"});
        InputStream in = new ByteArrayInputStream("conf/test_bq_very_simple.toml".getBytes(StandardCharsets.UTF_8));
        InputStream systemIn = System.in;
        try {
            System.setIn(in);
            ConfigExecutor.main(new String[0]);
        } finally {
            System.setIn(systemIn);
        }
        assertThrowsExactly(IllegalArgumentException.class, () -> ConfigExecutor.main(new String[]{"target/"}));
    }

}
