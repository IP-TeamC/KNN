package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.Neuron;

import java.util.LinkedList;
import java.util.List;

public class NetworkPaths {

    private final Connection[][][][][] allPaths;

    public NetworkPaths(Network network) {
        allPaths = new Connection[network.denseLayers.length][][][][];
        Neuron[] outputNeurons = network.denseLayers[network.denseLayers.length - 1].neurons;
        for (int layer = 0; layer < network.denseLayers.length; layer++) {
            Neuron[] neurons = network.denseLayers[layer].neurons;
            allPaths[layer] = new Connection[neurons.length][][][];
            for (int start = 0; start < neurons.length; start++) {
                Neuron neuron = neurons[start];
                allPaths[layer][start] = new Connection[outputNeurons.length][][];
                for (int output = 0; output < outputNeurons.length; output++) {
                    Connection[][] paths = getAllPaths((DenseNeuron) neuron, (DenseNeuron) outputNeurons[output])
                            .stream().map(path -> path.toArray(Connection[]::new)).toArray(Connection[][]::new);
                    allPaths[layer][start][output] = paths;
                }
            }
        }
    }

    public double pathsAdjustment(int startLayer, int startNeuron, int outputNeuron) {
        Connection[][] paths = allPaths[startLayer][startNeuron][outputNeuron];
        double pathsAdjustment = 1;

        for (Connection[] path : paths) {
            Connection pathConn = path[path.length - 1];
            pathsAdjustment *= pathConn.weight;

            for (int i = path.length - 2; i >= 0; i--) {
                pathsAdjustment *= pathConn.inputNeuron.output();
                pathConn = path[i];
                pathsAdjustment *= pathConn.weight;
            }
        }

        return pathsAdjustment;
    }

    private List<List<Connection>> getAllPaths(DenseNeuron startNeuron, DenseNeuron endNeuron) {
        if (startNeuron == endNeuron) {
            return new LinkedList<>();
        }

        List<List<Connection>> paths = new LinkedList<>();
        for (int i = 0; i < endNeuron.incoming.length; i++) {
            Connection conn = endNeuron.incoming[i];
            if (!conn.guard) continue;

            if (conn.inputNeuron == startNeuron) {
                List<Connection> simplePath = new LinkedList<>();
                simplePath.add(conn);
                return List.of(simplePath);
            } else if (conn.inputNeuron instanceof DenseNeuron dn) {
                for (List<Connection> path : getAllPaths(startNeuron, dn)) {
                    path.add(conn);
                    paths.add(path);
                }
            } else {
                return new LinkedList<>();
            }
        }

        return paths;
    }

}
