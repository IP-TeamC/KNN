package de.fhdw.knn.trainer.stop;

import java.util.LinkedList;
import java.util.List;

public class EarlyStopping implements StopFunction {

    public double minDelta;
    public int patience;
    public List<Double> losses = new LinkedList<>();

    private int epochsWaited = 0;

    @Override
    public boolean isFinished(double loss) {
        if (losses.getLast() - loss < minDelta) {
            epochsWaited += 1;
            losses.add(loss);
            return epochsWaited >= patience;
        } else {
            epochsWaited = 0;
            losses.add(loss);
            return false;
        }
    }

}
