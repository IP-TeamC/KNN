package de.fhdw.knn.config.network;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConfNetworkTest {

    @Test
    void testConfNetworkCreateWithLayers() {
        ConfLayer layer1 = new ConfLayer();
        layer1.neurons = 4;
        layer1.activationFunction = "LINEAR";

        ConfNetwork conf = new ConfNetwork();
        conf.seed = 42;
        conf.weightInitializer = "GLOROT_UNIFORM";
        conf.layer = new ConfLayer[]{layer1};

        DataSet ds = new DataSet(new double[][]{{1.0, 2.0, 3.0}}, new double[][]{{0.0}} );
        Network network = conf.create(ds);
        assertNotNull(network);
    }

    @Test
    void testConfNetworkCreateWithImportFile() {
        ConfNetwork conf = new ConfNetwork();
        conf.importFile = "models/test_conf.knn";

        DataSet ds = new DataSet(new double[][]{{1.0, 2.0, 3.0}}, new double[][]{{0.0}} );
        Network network = conf.create(ds);
        assertNotNull(network);
    }

    @Test
    void testConfNetworkCreateWithImportFileAndExtraConfigThrow() {

        for (int i = 0; i < 3; i++) {
            ConfNetwork conf = new ConfNetwork();
            conf.importFile = "models/test_conf.knn";

            switch (i) {
                case 0:
                    conf.seed = 42;
                    break;
                case 1:
                    conf.weightInitializer = "GLOROT_UNIFORM";
                    break;
                case 2:
                    conf.layer = new ConfLayer[]{new ConfLayer()};
                    break;
            }

            DataSet ds = new DataSet(new double[][]{{1.0, 2.0, 3.0}}, new double[][]{{0.0}} );
            assertThrows(IllegalArgumentException.class, () -> conf.create(ds));
        }

    }
}
