package de.fhdw.knn.util;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.visualization.HeatmapData;

public class FakeHeatmapData extends HeatmapData {
    public FakeHeatmapData(Network network) {
        super(network);
    }

    @Override
    public double[][] buildFullWeightMatrix() {
        double[][] m = new double[2][2];
        m[0][0] = 0.1;
        m[0][1] = 0.2;
        m[1][0] = 0.3;
        m[1][1] = 0.4;
        return m;
    }

    @Override
    public String[] getNeuronLabels() {
        return new String[]{"A", "B"};
    }
}
