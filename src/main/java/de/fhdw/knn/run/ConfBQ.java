package de.fhdw.knn.run;

import de.fhdw.knn.config.Config;

import java.io.IOException;

class ConfBQ {

    public static void main(String[] args) throws IOException {
        Config config = Config.read("conf/bq.toml");
        config.execute();
    }

}
