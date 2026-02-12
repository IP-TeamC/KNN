package de.fhdw.knn.run;

import de.fhdw.knn.config.Config;

import java.io.IOException;
import java.util.Scanner;

public class ConfigExecutor {

    public static void main(String[] args) throws IOException {
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
