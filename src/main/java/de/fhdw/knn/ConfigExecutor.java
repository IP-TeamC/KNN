package de.fhdw.knn;

import de.fhdw.knn.config.Config;
import lombok.SneakyThrows;

import java.util.Scanner;

/**
 * Main-Klasse zur Ausführung der TOML-Konfigurationen
 *
 * @see Config
 */
public class ConfigExecutor {

    /**
     * Über System.in wird eine TOML-Konfigurationsdatei angegeben (relativ zum Pfad ./conf/),
     * wobei die Angabe der Dateiendung .toml optional und automatisch ergänzt wird.
     * Die Konfiguration wird ausgelesen und ausgeführt.
     *
     * @param args Kommandozeilen-Argumente werden nicht ausgewertet
     * @see Config
     */
    @SneakyThrows
    public static void main(String[] args) {
        System.out.print("Config (.toml-Dateiendung optional): ./conf/");
        try (Scanner scanner = new Scanner(System.in)) {
            String file = "./conf/" + scanner.nextLine();
            if (!file.endsWith(".toml")) {
                file += ".toml";
            }

            System.out.printf("Führe Config '%s' aus...%n", file);
            Config config = Config.read(file);
            config.execute();
        }
    }

}
