package de.fhdw.knn;

import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
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
        // Erwarte bis hier keinen Fehler
        assertThrowsExactly(IllegalArgumentException.class, () -> ConfigExecutor.main(new String[]{"target/"}));
    }

    @Generated("GitHub Copilot")
    @Test
    public void testMainAcceptsTomlExtensionWithoutAppendingTwice() {
        String out = captureStdout(() -> ConfigExecutor.main(new String[]{"conf/test_bq_very_simple.toml"}));
        assertTrue(out.contains("conf/test_bq_very_simple.toml") || out.contains("conf\\test_bq_very_simple.toml"), "Aufgelöster Pfad sollte die übergebene Datei enthalten");
        assertFalse(out.contains(".toml.toml"), "Dateiendung .toml darf nicht doppelt ergänzt werden");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testResolveConfigPrefersDirectPathOverConfFallback() throws Exception {
        Path source = Path.of("conf/test_bq_very_simple.toml");
        Path direct = Path.of("test_bq_very_simple.toml");
        assertFalse(Files.exists(direct), "Test erwartet, dass die temporäre Datei noch nicht existiert");

        Files.copy(source, direct);
        try {
            String out = captureStdout(() -> ConfigExecutor.main(new String[]{"test_bq_very_simple"}));
            assertTrue(out.contains("test_bq_very_simple.toml"), "Ausgabe sollte den direkten Dateipfad enthalten");
            assertFalse(out.contains("conf/test_bq_very_simple.toml"),
                    "Direkter Pfad muss Priorität vor ./conf-Fallback haben");
        } finally {
            Files.deleteIfExists(direct);
        }
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
