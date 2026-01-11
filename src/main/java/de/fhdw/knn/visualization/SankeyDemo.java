package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Importer;
import eu.hansolo.fx.charts.SankeyPlot;
import eu.hansolo.fx.charts.data.PlotItem;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class SankeyDemo extends Application {

    @Override
    public void start(Stage primaryStage) {
        // -------------------- Output Layer --------------------
        PlotItem o_1 = new PlotItem("O-1", 1, Color.ORANGERED, 0);

        // -------------------- Hidden Layer 2 --------------------
        PlotItem h2_1 = new PlotItem("H2-1", 1, Color.RED, 1);
        PlotItem h2_2 = new PlotItem("H2-2", 1, Color.ORANGE, 1);
        PlotItem h2_3 = new PlotItem("H2-3", 1, Color.LIGHTGREEN, 1);
        PlotItem h2_4 = new PlotItem("H2-4", 1, Color.GOLD, 1);

        // -------------------- Hidden Layer 1 --------------------
        PlotItem h1_1 = new PlotItem("H1-1", 1, Color.GOLD, 2);
        PlotItem h1_2 = new PlotItem("H1-2", 1, Color.LIGHTGREEN, 2);
        PlotItem h1_3 = new PlotItem("H1-3", 1, Color.LIGHTGREEN, 2);
        PlotItem h1_4 = new PlotItem("H1-4", 1, Color.RED, 2);

        // -------------------- Input Layer --------------------
        PlotItem i_1 = new PlotItem("I-1", 1, Color.GOLD, 3);
        PlotItem i_2 = new PlotItem("I-2", 1, Color.RED, 3);
        PlotItem i_3 = new PlotItem("I-3", 1, Color.LIGHTGREEN, 3);
        PlotItem i_4 = new PlotItem("I-4", 1, Color.LIGHTGREEN, 3);
        PlotItem i_5 = new PlotItem("I-5", 1, Color.ORANGERED, 3);

        // -------------------- Verbindungen Output -> Hidden2 --------------------
        o_1.addToOutgoing(h2_1, 0.2);
        o_1.addToOutgoing(h2_2, 0.5);
        o_1.addToOutgoing(h2_3, 0.1);
        o_1.addToOutgoing(h2_4, 0.2);

        // -------------------- Verbindungen Hidden2 -> Hidden1 --------------------
        h2_1.addToOutgoing(h1_1, 0.2);
        h2_1.addToOutgoing(h1_2, 0.2);
        h2_1.addToOutgoing(h1_3, 0.3);
        h2_1.addToOutgoing(h1_4, 0.3);

        h2_2.addToOutgoing(h1_1, 0.8);
        h2_2.addToOutgoing(h1_2, 0.2);
        // h2_2.addToOutgoing(h1_3, 0.0);
        // h2_2.addToOutgoing(h1_4, 0.0);

        h2_3.addToOutgoing(h1_1, 0.4);
        h2_3.addToOutgoing(h1_2, 0.3);
        // h2_3.addToOutgoing(h1_3, 0.0);
        h2_3.addToOutgoing(h1_4, 0.2);

        h2_4.addToOutgoing(h1_1, 0.3);
        // h2_4.addToOutgoing(h1_2, 0.0);
        h2_4.addToOutgoing(h1_3, 0.3);
        h2_4.addToOutgoing(h1_4, 0.4);

        // -------------------- Verbindungen Hidden1 -> Input --------------------
        h1_1.addToOutgoing(i_1, 0.2);
        h1_1.addToOutgoing(i_2, 0.1);
        h1_1.addToOutgoing(i_3, 0.6);
        h1_1.addToOutgoing(i_4, 0.1);
        //  h1_1.addToOutgoing(i_5, 0.0);

        // h1_2.addToOutgoing(i_1, 0.0);
        h1_2.addToOutgoing(i_2, 0.2);
        // h1_2.addToOutgoing(i_3, 0.0);
        h1_2.addToOutgoing(i_4, 0.8);
        // h1_2.addToOutgoing(i_5, 0.0);

        h1_3.addToOutgoing(i_1, 0.3);
        h1_3.addToOutgoing(i_2, 0.2);
        h1_3.addToOutgoing(i_3, 0.1);
        h1_3.addToOutgoing(i_4, 0.2);
        h1_3.addToOutgoing(i_5, 0.2);

        h1_4.addToOutgoing(i_1, 0.1);
        h1_4.addToOutgoing(i_2, 0.1);
        h1_4.addToOutgoing(i_3, 0.5);
        h1_4.addToOutgoing(i_4, 0.3);
        // h1_4.addToOutgoing(i_5, 0.0);

        // -------------------- SankeyPlot --------------------
        SankeyPlot sankey = new SankeyPlot();
        sankey.setItems(
                o_1,
                h2_1, h2_2, h2_3, h2_4,
                h1_1, h1_2, h1_3, h1_4,
                i_1, i_2, i_3, i_4, i_5
        );

        sankey.setStreamFillMode(SankeyPlot.StreamFillMode.GRADIENT);
        sankey.setShowFlowDirection(false);

        // -------------------- Szene --------------------
        StackPane root = new StackPane(sankey);
        Scene scene = new Scene(root, 1000, 700);

        primaryStage.setTitle("Künstliches Neuronales Netz Demo");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}