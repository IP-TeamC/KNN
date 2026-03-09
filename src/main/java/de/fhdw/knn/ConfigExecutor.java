package de.fhdw.knn;

import de.fhdw.knn.config.Config;
import lombok.SneakyThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * Main-Klasse zur Ausführung der TOML-Konfigurationen
 *
 * @see Config
 */
public class ConfigExecutor {

    /**
     * Startpunkt der Anwendung.
     *
     * <p>Die Konfigurationsdatei kann entweder als Kommandozeilenargument übergeben oder
     * interaktiv über {@code System.in} eingegeben werden.
     *
     * <p>Der Pfad zur Konfigurationsdatei kann sowohl absolut als auch relativ zum Pfad {@code ./conf/} angegeben werden,
     * wobei die Angabe der Dateiendung {@code .toml} optional ist und automatisch ergänzt wird.
     *
     * <p>Die Konfiguration wird ausgelesen und ausgeführt.
     *
     * @param args Dateipfad zur Konfigurationsdatei (optional)
     * @see Config
     */
    @SneakyThrows
    public static void main(String[] args) {

        String filename;

        if (args.length > 0) {
            filename = args[0];
        }
        else {
            System.out.print("Config (.toml-Dateiendung optional): ./conf/");
            try (Scanner scanner = new Scanner(System.in)) {
                filename = scanner.nextLine();
            }
        }

        filename = addFileExtension(filename);
        Path confPath = resolveConfigFile(filename);

        System.out.printf("Führe Config '%s' aus...%n", confPath);
        Config config = Config.read(confPath.toString());
        config.execute();
    }

    /**
     * Ergänzt die Dateiendung {@code .toml} an den Dateinamen, falls diese nicht bereits vorhanden ist.
     *
     * @param filename Dateiname der Konfigurationsdatei
     * @return Dateiname mit {code .toml} als Dateiendung
     */
    private static String addFileExtension(String filename) {
        if (filename == null) return null;

        if (!filename.endsWith(".toml")) {
            filename += ".toml";
        }

        return filename;
    }

    /**
     * Löst den Dateipfad zur angegebenen Konfigurationsdatei auf. Die Methode prüft zunächst, ob die Datei
     * unter dem angegebenen Pfad existiert, und versucht andernfalls, den Dateipfad relativ zum Verzeichnis {@code ./conf}
     * aufzulösen. Wenn die Datei an keinem der beiden Orte gefunden werden kann, wird das Programm mit {@code System.exit(1)} beendet.
     *
     * @param filename der Name oder Pfad der aufzulösenden Konfigurationsdatei
     * @return der aufgelöste {@link Path} der Konfigurationsdatei
     */
    private static Path resolveConfigFile(String filename) {
        Path confPath = Path.of(filename);

        if (!Files.isRegularFile(confPath)) {
            confPath = Path.of("./conf", filename);
        }

        if (!Files.isRegularFile(confPath)) {
            System.err.println("Konfigurationsdatei nicht gefunden: " + confPath);
            System.exit(1);
        }

        return confPath;
    }
}
