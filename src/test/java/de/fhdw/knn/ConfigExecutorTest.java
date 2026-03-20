package de.fhdw.knn;

import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

public class ConfigExecutorTest {

    @Test
    public void testRun() {
        ConfigExecutor.main(new String[]{"test/bq_simple"});
        InputStream in = new ByteArrayInputStream("conf/test/bq_very_simple.toml".getBytes(StandardCharsets.UTF_8));
        InputStream systemIn = System.in;
        try {
            System.setIn(in);
            ConfigExecutor.main(new String[0]);
        } finally {
            System.setIn(systemIn);
        }
        // Erwarte bis hier keinen Fehler
        assertThrowsExactly(IllegalArgumentException.class, () -> ConfigExecutor.main(new String[]{"target/"}));
    }

    @Generated("GitHub Copilot")
    @Test
    public void testMainAcceptsTomlExtensionWithoutAppendingTwice() {
        String out = captureStdout(() -> ConfigExecutor.main(new String[]{"conf/test/bq_very_simple.toml"}));
        assertTrue(out.contains("conf/test/bq_very_simple.toml") || out.contains("conf\\test\\bq_very_simple.toml"), "Aufgelöster Pfad sollte die übergebene Datei enthalten");
        assertFalse(out.contains(".toml.toml"), "Dateiendung .toml darf nicht doppelt ergänzt werden");
    }

    @Generated("GitHub Copilot")
    private String captureStdout(Runnable action) {
        PrintStream oldOut = System.out;
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(buffer, true, StandardCharsets.UTF_8));
            action.run();
        } finally {
            System.setOut(oldOut);
        }
        return buffer.toString(StandardCharsets.UTF_8);
    }

}
