package de.fhdw.knn.trainer;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.neuron.Connection;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.loss.MeanSquaredError;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.IntStream;

public class Trainer {

    public int maxEpochs;
    public Network network;
    public LossFunction lossFunction = MeanSquaredError.DEFAULT;
    public StopFunction stopFunction;
    public OptimizationFunction optimizationFunction;
    public double learningRate;

    public Trainer(Network network, int maxEpochs, double learningRate) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputs, double[][] outputs) {
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            System.out.println("Epoche: " + epoch);
            trainEpoch(inputs, outputs);
        }
    }

    private void trainEpoch(double[][] inputs, double[][] outputs) {
        double totalLoss = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] predictions = network.feedForward(inputs[i]);
            totalLoss += lossFunction.loss(outputs[i], predictions);
            double derivedLoss = lossFunction.derivedLoss(outputs[i], predictions);
            // System.out.println("derived loss: " + derivedLoss);
            adjust(derivedLoss);
        }
        System.out.println("Loss: " + (totalLoss / inputs.length));
    }

    private void adjust(double derivedLoss) {
        double[][][] adjustmentsWeight = new double[network.denseLayers.length][][];
        double[][] adjustmentsBias = new double[network.denseLayers.length][];

        DenseNeuron[] outputNeurons = network.denseLayers[network.denseLayers.length - 1].neurons;
        for (int globalLayer = 0; globalLayer < network.denseLayers.length; globalLayer++) {
            DenseNeuron[] startNeurons = network.denseLayers[globalLayer].neurons;
            adjustmentsWeight[globalLayer] = new double[startNeurons.length][];
            adjustmentsBias[globalLayer] = new double[startNeurons.length];

            int layer = globalLayer;
            IntStream.range(0, startNeurons.length).parallel().forEach(start -> {
                Connection[] conns = startNeurons[start].incoming;
                adjustmentsWeight[layer][start] = new double[conns.length];

                for (int output = 0; output < outputNeurons.length; output++) {
                    List<List<Integer>> startOutputPaths = getAllPaths(startNeurons[start], outputNeurons[output]);
                    double adjustment = learningRate
                            * derivedLoss
                            * outputNeurons[output].outputDerivedCache;
                    if (layer != network.denseLayers.length - 1) {
                        adjustment *= startNeurons[start].outputDerivedCache;
                    }

                    double pathsAdjustment = 1;
                    for (List<Integer> path : startOutputPaths) {
                        Connection pathConn = outputNeurons[output].incoming[path.getLast()];
                        double pathAdjustment = pathConn.weight;


                        for (int i = path.size() - 2; i >= 0; i--) {
                            pathAdjustment *= pathConn.inputNeuron.output();
                            pathConn = ((DenseNeuron) pathConn.inputNeuron).incoming[path.get(i)];
                            pathsAdjustment *= pathConn.weight;
                        }

                        pathsAdjustment *= pathAdjustment;
                    }
                    adjustment *= pathsAdjustment;
                    adjustmentsBias[layer][start] = adjustment;

                    for (int conn = 0; conn < conns.length; conn++) {
                        double adjustmentWeight = adjustment * conns[conn].inputNeuron.output();
                        adjustmentsWeight[layer][start][conn] = adjustmentWeight;
                    }
                }
            });
        }

        for (int layer = 0; layer < network.denseLayers.length; layer++) {
            DenseNeuron[] neurons = network.denseLayers[layer].neurons;
            for (int neuron = 0; neuron < neurons.length; neuron++) {
                Connection[] conns = neurons[neuron].incoming;
                for (int conn = 0; conn < conns.length; conn++) {
                    conns[conn].weight -= adjustmentsWeight[layer][neuron][conn];
                }
                neurons[neuron].bias -= adjustmentsBias[layer][neuron];
            }
        }

//        System.out.println("adjustment h1:");
//        System.out.println(adjustmentsWeight[0][0][0]);
//        System.out.println(adjustmentsWeight[0][0][1]);
//        System.out.println(adjustmentsBias[0][0]);
//
//        System.out.println("adjustment h2:");
//        System.out.println(adjustmentsWeight[0][1][0]);
//        System.out.println(adjustmentsWeight[0][1][1]);
//        System.out.println(adjustmentsBias[0][1]);
//
//        System.out.println("adjustment output:");
//        System.out.println(adjustmentsWeight[1][0][0]);
//        System.out.println(adjustmentsWeight[1][0][1]);
//        System.out.println(adjustmentsBias[1][0]);
    }

    private List<List<Integer>> getAllPaths(DenseNeuron startNeuron, DenseNeuron endNeuron) {
        if (startNeuron == endNeuron) {
            return new LinkedList<>();
        }

        List<List<Integer>> paths = new LinkedList<>();
        for (int i = 0; i < endNeuron.incoming.length; i++) {
            Connection conn = endNeuron.incoming[i];
            if (conn.inputNeuron == startNeuron) {
                List<Integer> simplePath = new LinkedList<>();
                simplePath.add(i);
                return List.of(simplePath);
            } else if (conn.inputNeuron instanceof DenseNeuron dn) {
                for (List<Integer> path : getAllPaths(startNeuron, dn)) {
                    path.add(i);
                    paths.add(path);
                }
            } else {
                return new LinkedList<>();
            }
        }
        return paths;
    }

}
